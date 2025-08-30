// Natural Language Processing Benchmark
// Comprehensive NLP model benchmarks including BERT, GPT, and Transformer models

`ifndef NLP_BENCHMARK_SV
`define NLP_BENCHMARK_SV

class nlp_benchmark extends ai_benchmark_base;
    
    // NLP specific parameters
    string nlp_task = "text_classification";  // text_classification, question_answering, language_modeling, translation
    string dataset_name = "GLUE";
    int max_sequence_length = 512;
    int vocab_size = 30522;  // BERT vocab size
    int num_attention_heads = 12;
    int hidden_size = 768;
    int num_layers = 12;
    real dropout_rate = 0.1;
    
    // NLP metrics
    real f1_score = 0.0;
    real bleu_score = 0.0;
    real rouge_score = 0.0;
    real perplexity = 0.0;
    real exact_match = 0.0;
    
    // Model configurations for NLP
    typedef struct {
        string model_name;
        int max_seq_len;
        int vocab_size;
        int hidden_size;
        int num_layers;
        int num_heads;
        longint parameters;
        longint flops_per_token;
        real benchmark_score;
        string benchmark_metric;
    } nlp_model_config_t;
    
    nlp_model_config_t nlp_configs[model_type_e];
    
    // Token sequences for processing
    int input_tokens[][];
    int attention_masks[][];
    int token_type_ids[][];
    int output_tokens[][];
    int expected_tokens[][];
    
    `uvm_object_utils_begin(nlp_benchmark)
        `uvm_field_string(nlp_task, UVM_ALL_ON)
        `uvm_field_string(dataset_name, UVM_ALL_ON)
        `uvm_field_int(max_sequence_length, UVM_ALL_ON)
        `uvm_field_int(vocab_size, UVM_ALL_ON)
        `uvm_field_int(num_attention_heads, UVM_ALL_ON)
        `uvm_field_int(hidden_size, UVM_ALL_ON)
        `uvm_field_int(num_layers, UVM_ALL_ON)
        `uvm_field_real(f1_score, UVM_ALL_ON)
        `uvm_field_real(bleu_score, UVM_ALL_ON)
        `uvm_field_real(exact_match, UVM_ALL_ON)
    `uvm_object_utils_end
    
    function new(string name = "nlp_benchmark");
        super.new(name);
        initialize_nlp_configs();
    endfunction
    
    virtual function string get_benchmark_name();
        return $sformatf("NLP-%s-%s-%s", config.model_type.name(), nlp_task, dataset_name);
    endfunction
    
    virtual function void configure_benchmark(benchmark_config_t cfg);
        config = cfg;
        config.benchmark_type = NATURAL_LANGUAGE_PROCESSING;
        
        // Apply model-specific configuration
        if (nlp_configs.exists(config.model_type)) begin
            nlp_model_config_t model_cfg = nlp_configs[config.model_type];
            max_sequence_length = model_cfg.max_seq_len;
            vocab_size = model_cfg.vocab_size;
            hidden_size = model_cfg.hidden_size;
            num_layers = model_cfg.num_layers;
            num_attention_heads = model_cfg.num_heads;
            config.target_accuracy = model_cfg.benchmark_score;
        end else begin
            `uvm_warning(get_type_name(), $sformatf("Unknown NLP model: %s, using defaults", config.model_type.name()))
        end
        
        // Configure task-specific parameters
        configure_nlp_task();
        
        is_initialized = 1;
    endfunction
    
    virtual function bit initialize_benchmark();
        if (!is_initialized) return 0;
        
        `uvm_info(get_type_name(), $sformatf("Initializing %s NLP benchmark for %s task", 
                 config.model_type.name(), nlp_task), UVM_MEDIUM)
        
        // Load NLP dataset
        if (!load_nlp_dataset()) begin
            `uvm_error(get_type_name(), "Failed to load NLP dataset")
            return 0;
        end
        
        return 1;
    endfunction
    
    virtual task run_benchmark();
        `uvm_info(get_type_name(), $sformatf("Running NLP %s task on %0d samples", 
                 nlp_task, config.num_samples), UVM_MEDIUM)
        
        // Process text samples in batches
        int num_batches = config.num_samples / config.batch_size;
        for (int batch = 0; batch < num_batches; batch++) begin
            process_nlp_batch(batch);
            
            // Progress reporting
            if (batch % 50 == 0) begin
                `uvm_info(get_type_name(), $sformatf("Processed %0d/%0d batches", batch, num_batches), UVM_MEDIUM)
            end
        end
        
        results.total_samples_processed = num_batches * config.batch_size;
    endtask
    
    virtual function void analyze_results();
        calculate_nlp_metrics();
        calculate_performance_metrics();
        calculate_nlp_specific_metrics();
    endfunction
    
    // Initialize NLP model configurations
    virtual function void initialize_nlp_configs();
        // BERT models
        nlp_configs[BERT_BASE] = '{
            model_name: "BERT-Base",
            max_seq_len: 512,
            vocab_size: 30522,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            parameters: 110000000,
            flops_per_token: 22500000,
            benchmark_score: 84.3,  // GLUE score
            benchmark_metric: "GLUE"
        };
        
        nlp_configs[BERT_LARGE] = '{
            model_name: "BERT-Large",
            max_seq_len: 512,
            vocab_size: 30522,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            parameters: 340000000,
            flops_per_token: 70000000,
            benchmark_score: 86.6,  // GLUE score
            benchmark_metric: "GLUE"
        };
        
        // GPT models
        nlp_configs[GPT2_SMALL] = '{
            model_name: "GPT-2 Small",
            max_seq_len: 1024,
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            parameters: 117000000,
            flops_per_token: 24000000,
            benchmark_score: 29.4,  // Perplexity on WikiText-103
            benchmark_metric: "Perplexity"
        };
        
        nlp_configs[GPT2_MEDIUM] = '{
            model_name: "GPT-2 Medium",
            max_seq_len: 1024,
            vocab_size: 50257,
            hidden_size: 1024,
            num_layers: 24,
            num_heads: 16,
            parameters: 345000000,
            flops_per_token: 71000000,
            benchmark_score: 26.4,  // Perplexity on WikiText-103
            benchmark_metric: "Perplexity"
        };
        
        // Transformer models
        nlp_configs[TRANSFORMER_BASE] = '{
            model_name: "Transformer Base",
            max_seq_len: 512,
            vocab_size: 32000,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            parameters: 65000000,
            flops_per_token: 13000000,
            benchmark_score: 27.3,  // BLEU score on WMT14 EN-DE
            benchmark_metric: "BLEU"
        };
        
        // LSTM models
        nlp_configs[LSTM_SMALL] = '{
            model_name: "LSTM Small",
            max_seq_len: 256,
            vocab_size: 10000,
            hidden_size: 256,
            num_layers: 2,
            num_heads: 1,  // N/A for LSTM
            parameters: 5000000,
            flops_per_token: 2000000,
            benchmark_score: 92.5,  // Accuracy on IMDB
            benchmark_metric: "Accuracy"
        };
        
        nlp_configs[LSTM_LARGE] = '{
            model_name: "LSTM Large",
            max_seq_len: 512,
            vocab_size: 50000,
            hidden_size: 1024,
            num_layers: 4,
            num_heads: 1,  // N/A for LSTM
            parameters: 50000000,
            flops_per_token: 20000000,
            benchmark_score: 94.2,  // Accuracy on IMDB
            benchmark_metric: "Accuracy"
        };
    endfunction
    
    // Configure NLP task-specific parameters
    virtual function void configure_nlp_task();
        case (nlp_task)
            "text_classification": begin
                dataset_name = "IMDB";
                config.num_samples = 25000;
                config.num_classes = 2;  // Binary classification
                config.target_accuracy = 92.0;
            end
            "question_answering": begin
                dataset_name = "SQuAD";
                config.num_samples = 10833;
                config.target_accuracy = 88.5;  // F1 score
            end
            "language_modeling": begin
                dataset_name = "WikiText-103";
                config.num_samples = 3760;
                config.target_accuracy = 30.0;  // Perplexity (lower is better)
            end
            "translation": begin
                dataset_name = "WMT14-EN-DE";
                config.num_samples = 3003;
                config.target_accuracy = 27.0;  // BLEU score
            end
            "sentiment_analysis": begin
                dataset_name = "SST-2";
                config.num_samples = 872;
                config.num_classes = 2;
                config.target_accuracy = 93.5;
            end
            default: begin
                `uvm_warning(get_type_name(), $sformatf("Unknown NLP task: %s, using text_classification", nlp_task))
                nlp_task = "text_classification";
                configure_nlp_task();
            end
        endcase
    endfunction
    
    // Load NLP dataset
    virtual function bit load_nlp_dataset();
        `uvm_info(get_type_name(), $sformatf("Loading %s dataset for %s task", dataset_name, nlp_task), UVM_MEDIUM)
        
        // Generate synthetic NLP data
        generate_synthetic_nlp_data();
        
        return 1;
    endfunction
    
    // Generate synthetic NLP data
    virtual function void generate_synthetic_nlp_data();
        // Allocate token arrays
        input_tokens = new[config.batch_size];
        attention_masks = new[config.batch_size];
        token_type_ids = new[config.batch_size];
        expected_tokens = new[config.batch_size];
        
        for (int i = 0; i < config.batch_size; i++) begin
            // Generate random sequence length
            int seq_len = $urandom_range(10, max_sequence_length);
            
            // Allocate token sequences
            input_tokens[i] = new[seq_len];
            attention_masks[i] = new[seq_len];
            token_type_ids[i] = new[seq_len];
            
            // Generate input tokens
            for (int j = 0; j < seq_len; j++) begin
                input_tokens[i][j] = $urandom_range(1, vocab_size - 1);  // Avoid special tokens
                attention_masks[i][j] = 1;  // All tokens are real (not padding)
                token_type_ids[i][j] = (j < seq_len/2) ? 0 : 1;  // Simple segment split
            end
            
            // Generate expected outputs based on task
            case (nlp_task)
                "text_classification", "sentiment_analysis": begin
                    expected_tokens[i] = new[config.num_classes];
                    int true_class = $urandom_range(0, config.num_classes - 1);
                    for (int k = 0; k < config.num_classes; k++) begin
                        expected_tokens[i][k] = (k == true_class) ? 1 : 0;
                    end
                end
                "question_answering": begin
                    // For QA, expected output is start and end positions
                    expected_tokens[i] = new[2];
                    expected_tokens[i][0] = $urandom_range(0, seq_len/2);  // Start position
                    expected_tokens[i][1] = $urandom_range(expected_tokens[i][0], seq_len-1);  // End position
                end
                "language_modeling", "translation": begin
                    // For generation tasks, expected output is next token sequence
                    expected_tokens[i] = new[seq_len];
                    for (int k = 0; k < seq_len; k++) begin
                        expected_tokens[i][k] = $urandom_range(1, vocab_size - 1);
                    end
                end
                default: begin
                    expected_tokens[i] = new[1];
                    expected_tokens[i][0] = $urandom_range(0, config.num_classes - 1);
                end
            endcase
        end
        
        `uvm_info(get_type_name(), $sformatf("Generated synthetic NLP data: %0d sequences, max_len=%0d", 
                 config.batch_size, max_sequence_length), UVM_MEDIUM)
    endfunction
    
    // Process NLP batch
    virtual task process_nlp_batch(int batch_idx);
        time batch_start = $time;
        
        // Simulate tokenization (if not pre-tokenized)
        simulate_tokenization();
        
        // Simulate model inference
        simulate_nlp_inference();
        
        // Simulate post-processing
        simulate_nlp_postprocessing();
        
        time batch_end = $time;
        real batch_latency_ms = real'(batch_end - batch_start) / 1e6;
        
        // Update latency
        if (batch_latency_ms > results.latency_ms) begin
            results.latency_ms = batch_latency_ms;
        end
        
        // Update operation count
        if (nlp_configs.exists(config.model_type)) begin
            longint batch_operations = nlp_configs[config.model_type].flops_per_token * 
                                     max_sequence_length * config.batch_size;
            results.total_operations += batch_operations;
        end
    endtask
    
    // Simulate tokenization
    virtual task simulate_tokenization();
        // Tokenization time depends on sequence length and vocabulary size
        time tokenize_delay = $urandom_range(50, 200) * 1ns * max_sequence_length;
        #tokenize_delay;
    endtask
    
    // Simulate NLP model inference
    virtual task simulate_nlp_inference();
        time inference_delay;
        
        // Model-specific inference timing (scales with sequence length and model size)
        case (config.model_type)
            BERT_BASE: begin
                inference_delay = $urandom_range(100, 300) * 1ns * max_sequence_length;
            end
            BERT_LARGE: begin
                inference_delay = $urandom_range(300, 800) * 1ns * max_sequence_length;
            end
            GPT2_SMALL: begin
                inference_delay = $urandom_range(80, 250) * 1ns * max_sequence_length;
            end
            GPT2_MEDIUM: begin
                inference_delay = $urandom_range(250, 600) * 1ns * max_sequence_length;
            end
            TRANSFORMER_BASE: begin
                inference_delay = $urandom_range(60, 180) * 1ns * max_sequence_length;
            end
            LSTM_SMALL: begin
                inference_delay = $urandom_range(20, 80) * 1ns * max_sequence_length;
            end
            LSTM_LARGE: begin
                inference_delay = $urandom_range(80, 200) * 1ns * max_sequence_length;
            end
            default: begin
                inference_delay = $urandom_range(100, 300) * 1ns * max_sequence_length;
            end
        endcase
        
        // Scale by precision
        case (config.precision)
            INT8_QUANT: inference_delay = inference_delay / 4;
            FP16_HALF: inference_delay = inference_delay / 2;
            FP32_SINGLE: /* no change */;
            MIXED_PRECISION: inference_delay = inference_delay * 0.8;
        endcase
        
        #(inference_delay * config.batch_size);
        
        // Generate synthetic outputs
        generate_synthetic_nlp_outputs();
    endtask
    
    // Generate synthetic NLP outputs
    virtual function void generate_synthetic_nlp_outputs();
        output_tokens = new[config.batch_size];
        
        for (int i = 0; i < config.batch_size; i++) begin
            case (nlp_task)
                "text_classification", "sentiment_analysis": begin
                    // Classification logits
                    output_tokens[i] = new[config.num_classes];
                    real total_logit = 0.0;
                    
                    // Generate logits
                    for (int j = 0; j < config.num_classes; j++) begin
                        real logit = $urandom_range(0, 1000) / 100.0;
                        output_tokens[i][j] = $rtoi(logit * 100);
                        total_logit += logit;
                    end
                    
                    // Boost correct class based on target accuracy
                    if ($urandom_range(0, 99) < $rtoi(config.target_accuracy)) begin
                        int correct_class = find_expected_class(i);
                        if (correct_class >= 0) begin
                            output_tokens[i][correct_class] = $rtoi(total_logit * 150);
                        end
                    end
                end
                
                "question_answering": begin
                    // Start and end position logits
                    output_tokens[i] = new[2];
                    output_tokens[i][0] = $urandom_range(0, max_sequence_length - 1);  // Start pos
                    output_tokens[i][1] = $urandom_range(output_tokens[i][0], max_sequence_length - 1);  // End pos
                end
                
                "language_modeling", "translation": begin
                    // Generated token sequence
                    output_tokens[i] = new[max_sequence_length];
                    for (int k = 0; k < max_sequence_length; k++) begin
                        output_tokens[i][k] = $urandom_range(1, vocab_size - 1);
                    end
                end
                
                default: begin
                    output_tokens[i] = new[1];
                    output_tokens[i][0] = $urandom_range(0, vocab_size - 1);
                end
            endcase
        end
    endfunction
    
    // Find expected class for sample
    virtual function int find_expected_class(int sample_idx);
        if (expected_tokens[sample_idx].size() == 0) return -1;
        
        for (int i = 0; i < expected_tokens[sample_idx].size(); i++) begin
            if (expected_tokens[sample_idx][i] == 1) return i;
        end
        return 0;  // Default to first class
    endfunction
    
    // Simulate NLP post-processing
    virtual task simulate_nlp_postprocessing();
        // Post-processing includes softmax, beam search, etc.
        time postprocess_delay;
        
        case (nlp_task)
            "text_classification", "sentiment_analysis": begin
                postprocess_delay = $urandom_range(10, 50) * 1ns;  // Simple softmax
            end
            "question_answering": begin
                postprocess_delay = $urandom_range(50, 200) * 1ns;  // Position extraction
            end
            "language_modeling", "translation": begin
                postprocess_delay = $urandom_range(200, 1000) * 1ns;  // Beam search/sampling
            end
            default: begin
                postprocess_delay = $urandom_range(20, 100) * 1ns;
            end
        endcase
        
        #(postprocess_delay * config.batch_size);
    endtask
    
    // Calculate NLP-specific metrics
    virtual function void calculate_nlp_metrics();
        case (nlp_task)
            "text_classification", "sentiment_analysis": begin
                calculate_classification_accuracy();
            end
            "question_answering": begin
                calculate_qa_metrics();
            end
            "language_modeling": begin
                calculate_perplexity();
            end
            "translation": begin
                calculate_bleu_score();
            end
            default: begin
                calculate_classification_accuracy();
            end
        endcase
    endfunction
    
    // Calculate classification accuracy
    virtual function void calculate_classification_accuracy();
        int correct_predictions = 0;
        int total_predictions = config.batch_size;
        
        for (int i = 0; i < config.batch_size; i++) begin
            // Find predicted class (highest logit)
            int predicted_class = 0;
            int max_logit = output_tokens[i][0];
            for (int j = 1; j < output_tokens[i].size(); j++) begin
                if (output_tokens[i][j] > max_logit) begin
                    max_logit = output_tokens[i][j];
                    predicted_class = j;
                end
            end
            
            // Find expected class
            int expected_class = find_expected_class(i);
            
            if (predicted_class == expected_class) begin
                correct_predictions++;
            end
        end
        
        results.accuracy_top1 = real'(correct_predictions) / real'(total_predictions) * 100.0;
        f1_score = results.accuracy_top1;  // Simplified F1 calculation
    endfunction
    
    // Calculate QA metrics (simplified)
    virtual function void calculate_qa_metrics();
        int exact_matches = 0;
        real total_f1 = 0.0;
        
        for (int i = 0; i < config.batch_size; i++) begin
            // Check if predicted span matches expected span
            if (output_tokens[i].size() >= 2 && expected_tokens[i].size() >= 2) begin
                if (output_tokens[i][0] == expected_tokens[i][0] && 
                    output_tokens[i][1] == expected_tokens[i][1]) begin
                    exact_matches++;
                    total_f1 += 1.0;  // Perfect match gets F1=1.0
                end else begin
                    // Calculate overlap F1 (simplified)
                    real overlap_f1 = calculate_span_overlap_f1(output_tokens[i], expected_tokens[i]);
                    total_f1 += overlap_f1;
                end
            end
        end
        
        exact_match = real'(exact_matches) / real'(config.batch_size) * 100.0;
        f1_score = total_f1 / real'(config.batch_size) * 100.0;
        results.accuracy_top1 = f1_score;
    endfunction
    
    // Calculate span overlap F1 score
    virtual function real calculate_span_overlap_f1(int predicted[], int expected[]);
        if (predicted.size() < 2 || expected.size() < 2) return 0.0;
        
        int pred_start = predicted[0];
        int pred_end = predicted[1];
        int exp_start = expected[0];
        int exp_end = expected[1];
        
        // Calculate overlap
        int overlap_start = (pred_start > exp_start) ? pred_start : exp_start;
        int overlap_end = (pred_end < exp_end) ? pred_end : exp_end;
        
        if (overlap_end <= overlap_start) return 0.0;  // No overlap
        
        int overlap_len = overlap_end - overlap_start + 1;
        int pred_len = pred_end - pred_start + 1;
        int exp_len = exp_end - exp_start + 1;
        
        real precision = real'(overlap_len) / real'(pred_len);
        real recall = real'(overlap_len) / real'(exp_len);
        
        if (precision + recall == 0.0) return 0.0;
        
        return 2.0 * precision * recall / (precision + recall);
    endfunction
    
    // Calculate perplexity (simplified)
    virtual function void calculate_perplexity();
        // Simplified perplexity calculation
        real total_log_prob = 0.0;
        int total_tokens = 0;
        
        for (int i = 0; i < config.batch_size; i++) begin
            for (int j = 0; j < output_tokens[i].size(); j++) begin
                // Simulate log probability (in real implementation, would use actual model outputs)
                real log_prob = -$urandom_range(100, 1000) / 1000.0;  // Negative log prob
                total_log_prob += log_prob;
                total_tokens++;
            end
        end
        
        if (total_tokens > 0) begin
            real avg_log_prob = total_log_prob / real'(total_tokens);
            perplexity = $exp(-avg_log_prob);
        end else begin
            perplexity = 1000.0;  // High perplexity for no tokens
        end
        
        // For perplexity, lower is better, so we use inverse for accuracy
        results.accuracy_top1 = 100.0 / perplexity;
    endfunction
    
    // Calculate BLEU score (simplified)
    virtual function void calculate_bleu_score();
        real total_bleu = 0.0;
        
        for (int i = 0; i < config.batch_size; i++) begin
            // Simplified BLEU calculation (n-gram overlap)
            real sample_bleu = calculate_sample_bleu(output_tokens[i], expected_tokens[i]);
            total_bleu += sample_bleu;
        end
        
        bleu_score = total_bleu / real'(config.batch_size);
        results.accuracy_top1 = bleu_score;
    endfunction
    
    // Calculate BLEU score for single sample (simplified)
    virtual function real calculate_sample_bleu(int predicted[], int expected[]);
        if (predicted.size() == 0 || expected.size() == 0) return 0.0;
        
        // Count matching tokens (simplified 1-gram BLEU)
        int matches = 0;
        for (int i = 0; i < predicted.size(); i++) begin
            for (int j = 0; j < expected.size(); j++) begin
                if (predicted[i] == expected[j]) begin
                    matches++;
                    break;
                end
            end
        end
        
        real precision = real'(matches) / real'(predicted.size());
        
        // Apply brevity penalty
        real bp = 1.0;
        if (predicted.size() < expected.size()) begin
            bp = $exp(1.0 - real'(expected.size()) / real'(predicted.size()));
        end
        
        return bp * precision * 100.0;  // Convert to percentage
    endfunction
    
    // Calculate NLP-specific performance metrics
    virtual function void calculate_nlp_specific_metrics();
        // Tokens per second
        if (results.total_execution_time > 0) begin
            real seconds = real'(results.total_execution_time) / 1e9;
            real total_tokens = real'(max_sequence_length) * real'(results.total_samples_processed);
            real tokens_per_second = total_tokens / seconds;
            `uvm_info(get_type_name(), $sformatf("Tokens per second: %.2f", tokens_per_second), UVM_MEDIUM)
        end
        
        // Model efficiency metrics
        if (nlp_configs.exists(config.model_type)) begin
            nlp_model_config_t model_cfg = nlp_configs[config.model_type];
            
            // Score per parameter
            real mparams = real'(model_cfg.parameters) / 1e6;
            if (mparams > 0) begin
                real score_per_mparam = results.accuracy_top1 / mparams;
                `uvm_info(get_type_name(), $sformatf("Score per MParam: %.3f", score_per_mparam), UVM_MEDIUM)
            end
            
            // Score per GFLOP
            real gflops = real'(model_cfg.flops_per_token) * real'(max_sequence_length) / 1e9;
            if (gflops > 0) begin
                real score_per_gflop = results.accuracy_top1 / gflops;
                `uvm_info(get_type_name(), $sformatf("Score per GFLOP: %.3f", score_per_gflop), UVM_MEDIUM)
            end
        end
    endfunction
    
    // Override print results to include NLP metrics
    virtual function void print_results();
        super.print_results();
        
        `uvm_info(get_type_name(), "=== NLP SPECIFIC METRICS ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Task: %s", nlp_task), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Dataset: %s", dataset_name), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Max Sequence Length: %0d", max_sequence_length), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Vocabulary Size: %0d", vocab_size), UVM_LOW)
        
        case (nlp_task)
            "text_classification", "sentiment_analysis": begin
                `uvm_info(get_type_name(), $sformatf("F1 Score: %.2f%%", f1_score), UVM_LOW)
            end
            "question_answering": begin
                `uvm_info(get_type_name(), $sformatf("F1 Score: %.2f%%", f1_score), UVM_LOW)
                `uvm_info(get_type_name(), $sformatf("Exact Match: %.2f%%", exact_match), UVM_LOW)
            end
            "language_modeling": begin
                `uvm_info(get_type_name(), $sformatf("Perplexity: %.2f", perplexity), UVM_LOW)
            end
            "translation": begin
                `uvm_info(get_type_name(), $sformatf("BLEU Score: %.2f", bleu_score), UVM_LOW)
            end
        endcase
    endfunction
    
endclass : nlp_benchmark

`endif // NLP_BENCHMARK_SV