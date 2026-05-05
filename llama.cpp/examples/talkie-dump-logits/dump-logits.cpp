// Dump top-K logits at the position right after each prompt token, for one
// or more prompts. Used to validate the talkie GGUF graph against the
// reference PyTorch implementation (run on a CUDA box separately).
//
// JSON output (one record per prompt) looks like:
//   {
//     "prompt":     "The year 1930 was",
//     "token_ids":  [486, 607, 32, 5748, 48, 359],
//     "next_token": [{"id": ..., "logit": ...}, ...]   // top-K at last position
//   }
//
// Compatible with the matching `dump_reference_logits.py` script — both write
// the same fields, and `compare_logits.py` joins them.

#include "arg.h"
#include "common.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

std::string json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 2);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b";  break;
            case '\f': out += "\\f";  break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c & 0xff);
                    out += buf;
                } else {
                    out += c;
                }
        }
    }
    return out;
}

std::vector<std::string> read_lines(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        fprintf(stderr, "talkie-dump-logits: cannot open prompts file %s\n", path.c_str());
        std::exit(1);
    }
    std::vector<std::string> out;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty() || line[0] == '#') continue;
        // unescape \n so a single line can span multiple lines of prompt
        std::string un;
        un.reserve(line.size());
        for (size_t i = 0; i < line.size(); ++i) {
            if (line[i] == '\\' && i + 1 < line.size()) {
                char c = line[i+1];
                if      (c == 'n') { un += '\n'; ++i; }
                else if (c == 't') { un += '\t'; ++i; }
                else if (c == '\\') { un += '\\'; ++i; }
                else { un += line[i]; }
            } else {
                un += line[i];
            }
        }
        out.push_back(un);
    }
    return out;
}

}  // namespace

int main(int argc, char ** argv) {
    // ---- pre-parse our own CLI flags so common_params_parse doesn't reject them ----
    int top_k_out = 50;
    std::string prompts_file;
    std::string output_path = "talkie_logits.json";
    std::vector<char *> filtered_argv;
    filtered_argv.reserve(argc);
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--top-k-out" && i + 1 < argc) {
            top_k_out = std::atoi(argv[++i]);
        } else if (a == "--prompts-file" && i + 1 < argc) {
            prompts_file = argv[++i];
        } else if (a == "--out" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int)filtered_argv.size();

    common_params params;
    params.prompt = "The year 1930 was";
    params.n_predict = 0;          // we only need the prompt forward pass

    common_init();

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    std::vector<std::string> prompts;
    if (!prompts_file.empty()) {
        prompts = read_lines(prompts_file);
    } else {
        prompts.push_back(params.prompt);
    }
    if (prompts.empty()) {
        fprintf(stderr, "talkie-dump-logits: no prompts to run\n");
        return 1;
    }

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();
    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "talkie-dump-logits: failed to init model/context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (top_k_out > n_vocab) top_k_out = n_vocab;

    std::ofstream out(output_path);
    if (!out) {
        fprintf(stderr, "talkie-dump-logits: cannot write to %s\n", output_path.c_str());
        return 1;
    }

    out << "{\n  \"model\": \"" << json_escape(params.model.path) << "\",\n";
    out << "  \"top_k\": " << top_k_out << ",\n";
    out << "  \"results\": [\n";

    for (size_t pi = 0; pi < prompts.size(); ++pi) {
        const std::string & prompt = prompts[pi];

        // Each prompt gets a fresh KV state.
        llama_memory_clear(llama_get_memory(ctx), true);

        // Tokenize -- no automatic BOS, talkie's reference doesn't add one.
        std::vector<llama_token> tokens = common_tokenize(ctx, prompt, /*add_special=*/false, /*parse_special=*/true);
        if (tokens.empty()) {
            fprintf(stderr, "  prompt %zu: tokenized to 0 tokens, skipping\n", pi);
            continue;
        }

        llama_batch batch = llama_batch_init((int32_t)tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); ++i) {
            common_batch_add(batch, tokens[i], (int32_t)i, {0}, /*logits=*/false);
        }
        // Only request logits for the last token.
        batch.logits[batch.n_tokens - 1] = true;

        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "  prompt %zu: llama_decode failed\n", pi);
            llama_batch_free(batch);
            continue;
        }

        const float * logits = llama_get_logits_ith(ctx, batch.n_tokens - 1);
        if (logits == nullptr) {
            fprintf(stderr, "  prompt %zu: no logits available\n", pi);
            llama_batch_free(batch);
            continue;
        }

        // Top-K extraction.
        std::vector<std::pair<int, float>> indexed;
        indexed.reserve(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            indexed.emplace_back(i, logits[i]);
        }
        std::partial_sort(
            indexed.begin(), indexed.begin() + top_k_out, indexed.end(),
            [](const auto & a, const auto & b) { return a.second > b.second; });
        indexed.resize(top_k_out);

        out << "    {\n";
        out << "      \"prompt\": \"" << json_escape(prompt) << "\",\n";
        out << "      \"token_ids\": [";
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i) out << ", ";
            out << tokens[i];
        }
        out << "],\n";
        out << "      \"next_token\": [";
        for (size_t i = 0; i < indexed.size(); ++i) {
            if (i) out << ", ";
            out << "{\"id\": " << indexed[i].first
                << ", \"logit\": " << indexed[i].second << "}";
        }
        out << "]\n";
        out << "    }" << (pi + 1 == prompts.size() ? "" : ",") << "\n";

        // Print a tiny progress hint (the 3090 user can read this, low-vis safe).
        printf("  prompt %2zu (%zu tokens): top1 = %d (logit %.3f)\n",
               pi, tokens.size(), indexed[0].first, indexed[0].second);

        llama_batch_free(batch);
    }

    out << "  ]\n}\n";
    out.close();

    printf("wrote %s\n", output_path.c_str());
    return 0;
}
