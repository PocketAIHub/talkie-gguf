#include "models.h"

// Talkie 13B (vintage 1930-era LLM by Alec Radford et al.).
//
// Custom 40-layer / 40-head decoder-only transformer with:
//   * Scaleless RMSNorm everywhere (no learned scale parameter)
//   * Q/K RMSNorm after RoPE (also scaleless)
//   * Per-layer learned scalar that adds the *original* embedding (post a
//     pre-block RMSNorm) to the residual stream every layer (the "embed-skip"
//     connection — talkie's distinctive feature).
//   * SwiGLU MLP, NeoX-style RoPE.
//
// The converter folds the head-gain / attn-gain / mlp-gain / lm_head_gain
// scalars into the adjacent linear weights, so they aren't visible at this
// level — the only custom op left is the embed-skip add.

llm_build_talkie::llm_build_talkie(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    ggml_tensor * cur;
    ggml_tensor * inpL;

    inpL = build_inp_embd(model.tok_embd);

    ggml_tensor * inp_pos = build_inp_pos();

    auto * inp_attn = build_attn_inp_kv();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // Pre-block RMSNorm of the input embeddings. Reused as the "embed_skip"
    // source by every layer (talkie's distinctive feature).
    // ggml_dup to give e_x its own memory buffer that won't be reused by
    // later ops in the graph -- some prompt sizes (>=9 tokens) caused the
    // memory planner to alias e_x with later activations, producing NaN.
    ggml_tensor * e_x = build_norm(inpL, NULL, NULL, LLM_NORM_RMS, -1);
    e_x = ggml_dup(ctx0, e_x);
    cb(e_x, "embd_norm", -1);

    cur = ggml_dup(ctx0, e_x);

    for (int il = 0; il < n_layer; ++il) {
        // ---- self-attention block ----
        ggml_tensor * attn_residual = cur;

        // pre-attention norm (scaleless)
        ggml_tensor * attn_in = build_norm(cur, NULL, NULL, LLM_NORM_RMS, il);
        cb(attn_in, "attn_norm", il);

        ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, attn_in);
        cb(Qcur, "Qcur", il);

        ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, attn_in);
        cb(Kcur, "Kcur", il);

        ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, attn_in);
        cb(Vcur, "Vcur", il);

        Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head,    n_tokens);
        Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens);
        Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens);

        // RoPE (NeoX-style: pairs are (x[i], x[i + d/2])).
        // Talkie uses an inverse-direction rotation vs. ggml's standard NeoX
        // (its sin signs are flipped). Passing freq_scale = -1 negates theta,
        // which flips sin while leaving cos unchanged — matching talkie.
        const float talkie_freq_scale = -1.0f * freq_scale;
        Qcur = ggml_rope_ext(
                ctx0, Qcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, talkie_freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        Kcur = ggml_rope_ext(
                ctx0, Kcur, inp_pos, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, talkie_freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);

        // Q/K RMSNorm (scaleless, after RoPE — talkie convention).
        Qcur = build_norm(Qcur, NULL, NULL, LLM_NORM_RMS, il);
        cb(Qcur, "Qcur_normed", il);
        Kcur = build_norm(Kcur, NULL, NULL, LLM_NORM_RMS, il);
        cb(Kcur, "Kcur_normed", il);

        // Per-head Q gain (after RMSNorm — folding into wq doesn't work because
        // the post-RoPE Q RMSNorm cancels any pre-norm scaling).
        // attn_q_gain shape is [1, n_head]; Qcur shape is [head_dim, n_head, n_tokens];
        // ggml_mul broadcasts singleton dims, so the gain hits the right axis.
        ggml_tensor * q_gain_f32 = ggml_cast(ctx0, model.layers[il].attn_q_gain, GGML_TYPE_F32);
        Qcur = ggml_mul(ctx0, Qcur, q_gain_f32);
        cb(Qcur, "Qcur_gained", il);

        cb(Qcur, "Qcur", il);
        cb(Kcur, "Kcur", il);
        cb(Vcur, "Vcur", il);

        // SDPA + output projection (attn-gain already folded into wo).
        cur = build_attn(inp_attn,
                model.layers[il].wo, NULL, NULL,
                Qcur, Kcur, Vcur, nullptr, nullptr, nullptr,
                1.0f / sqrtf(float(n_embd_head)), il);
        cb(cur, "attn_out", il);

        cur = ggml_add(ctx0, cur, attn_residual);
        cb(cur, "after_attn_residual", il);

        // Slice down to the requested output ids on the last layer (an
        // optimization to skip MLP work on tokens whose logits aren't needed).
        // We must also slice e_x so the embed-skip add stays shape-compatible.
        if (il == n_layer - 1 && inp_out_ids) {
            cur = ggml_get_rows(ctx0, cur, inp_out_ids);
            e_x = ggml_get_rows(ctx0, e_x, inp_out_ids);
        }

        // ---- feed-forward block ----
        ggml_tensor * ffn_residual = cur;

        // pre-mlp norm (scaleless)
        ggml_tensor * mlp_in = build_norm(cur, NULL, NULL, LLM_NORM_RMS, il);
        cb(mlp_in, "ffn_norm", il);

        // SwiGLU (mlp-gain already folded into ffn_down).
        cur = build_ffn(mlp_in,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_residual);
        cb(cur, "after_ffn_residual", il);

        // Embed-skip term: cur += embed_skip[il] * e_x.
        // embed_skip_scale is stored as an [n_embd] tiled-scalar vector to avoid
        // a Metal broadcast bug with true [1] scalars at n_tokens >= 9.
        ggml_tensor * skip_scale_f32 = ggml_cast(ctx0, model.layers[il].embed_skip_scale, GGML_TYPE_F32);
        ggml_tensor * skip_term = ggml_mul(ctx0, e_x, skip_scale_f32);
        cur = ggml_add(ctx0, cur, skip_term);
        cb(cur, "after_embed_skip", il);

        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);
    }

    // Final scaleless RMSNorm.
    cur = build_norm(cur, NULL, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // lm_head (lm_head_gain already folded into model.output).
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
