// Recommendation System Benchmark
// Comprehensive recommendation model benchmarks

`ifndef RECOMMENDATION_BENCHMARK_SV
`define RECOMMENDATION_BENCHMARK_SV

class recommendation_benchmark extends ai_benchmark_base;
    
    // Recommendation specific parameters
    string rec_task = "collaborative_filtering";  // collaborative_filtering, content_based, hybrid
    string rec_dataset = "MovieLens";
    int num_users = 100000;
    int num_items = 10000;
    int embedding_dim = 128;
    int num_factors = 64;
    real learning_rate = 0.01;
    int top_k = 10;  // For top-K recommendation evaluation
    
    // Recommendation metrics
    real precision_at_k = 0.0;
    real recall_at_k = 0.0;
    real ndcg_at_k = 0.0;
    real map_score = 0.0;  // Mean Average Precision
    real auc_score = 0.0;  // Area Under Curve
    real rmse = 0.0;       // Root Mean Square Error
    real mae = 0.0;        // Mean Absolute Error
    
    // Model configurations for recommendation systems
    typedef struct {
        string model_name;
        int embedding_size;
        int hidden_layers;
        int hidden_units;
        longint parameters;
        longint flops_per_sample;
        real benchmark_score;
        string benchmark_metric;
    } rec_model_config_t;
    
    rec_model_config_t rec_configs[model_type_e];
    
    // Recommendation data structures
    typedef struct {
        int user_id;
        int item_id;
        real rating;
        real predicted_rating;
        real confidence;
    } interaction_t;
    
    interaction_t training_interactions[];
    interaction_t test_interactions[];
    interaction_t predicted_interactions[];
    
    // User and item features
    real user_embeddings[][];
    real item_embeddings[][];
    real user_features[][];
    real item_features[][];
    
    `uvm_object_utils_begin(recommendation_benchmark)
        `uvm_field_string(rec_task, UVM_ALL_ON)
        `uvm_field_string(rec_dataset, UVM_ALL_ON)
        `uvm_field_int(num_users, UVM_ALL_ON)
        `uvm_field_int(num_items, UVM_ALL_ON)
        `uvm_field_int(embedding_dim, UVM_ALL_ON)
        `uvm_field_int(top_k, UVM_ALL_ON)
        `uvm_field_real(precision_at_k, UVM_ALL_ON)
        `uvm_field_real(recall_at_k, UVM_ALL_ON)
        `uvm_field_real(ndcg_at_k, UVM_ALL_ON)
        `uvm_field_real(auc_score, UVM_ALL_ON)
        `uvm_field_real(rmse, UVM_ALL_ON)
    `uvm_object_utils_end
    
    function new(string name = "recommendation_benchmark");
        super.new(name);
        initialize_rec_configs();
    endfunction
    
    virtual function string get_benchmark_name();
        return $sformatf("Recommendation-%s-%s-%s", config.model_type.name(), rec_task, rec_dataset);
    endfunction
    
    virtual function void configure_benchmark(benchmark_config_t cfg);
        config = cfg;
        config.benchmark_type = RECOMMENDATION_SYSTEM;
        
        // Apply model-specific configuration
        if (rec_configs.exists(config.model_type)) begin
            rec_model_config_t model_cfg = rec_configs[config.model_type];
            embedding_dim = model_cfg.embedding_size;
            config.target_accuracy = model_cfg.benchmark_score;
        end else begin
            `uvm_warning(get_type_name(), $sformatf("Unknown recommendation model: %s, using defaults", config.model_type.name()))
        end
        
        // Configure dataset-specific parameters
        configure_rec_dataset();
        
        is_initialized = 1;
    endfunction
    
    virtual function bit initialize_benchmark();
        if (!is_initialized) return 0;
        
        `uvm_info(get_type_name(), $sformatf("Initializing %s recommendation benchmark on %s", 
                 config.model_type.name(), rec_dataset), UVM_MEDIUM)
        
        // Load recommendation dataset
        if (!load_rec_dataset()) begin
            `uvm_error(get_type_name(), "Failed to load recommendation dataset")
            return 0;
        end
        
        return 1;
    endfunction
    
    virtual task run_benchmark();
        `uvm_info(get_type_name(), $sformatf("Running recommendation %s on %0d interactions", 
                 rec_task, config.num_samples), UVM_MEDIUM)
        
        // Process recommendations in batches
        int num_batches = config.num_samples / config.batch_size;
        for (int batch = 0; batch < num_batches; batch++) begin
            process_recommendation_batch(batch);
            
            // Progress reporting
            if (batch % 100 == 0) begin
                `uvm_info(get_type_name(), $sformatf("Processed %0d/%0d batches", batch, num_batches), UVM_MEDIUM)
            end
        end
        
        results.total_samples_processed = num_batches * config.batch_size;
    endtask
    
    virtual function void analyze_results();
        calculate_recommendation_metrics();
        calculate_performance_metrics();
        calculate_rec_specific_metrics();
    endfunction
    
    // Initialize recommendation model configurations
    virtual function void initialize_rec_configs();
        // Wide & Deep model
        rec_configs[WIDE_DEEP] = '{
            model_name: "Wide & Deep",
            embedding_size: 128,
            hidden_layers: 3,
            hidden_units: 512,
            parameters: 50000000,
            flops_per_sample: 100000,
            benchmark_score: 0.85,  // AUC score
            benchmark_metric: "AUC"
        };
        
        // DeepFM model
        rec_configs[DEEP_FM] = '{
            model_name: "DeepFM",
            embedding_size: 64,
            hidden_layers: 3,
            hidden_units: 400,
            parameters: 30000000,
            flops_per_sample: 80000,
            benchmark_score: 0.87,  // AUC score
            benchmark_metric: "AUC"
        };
        
        // Neural Collaborative Filtering
        rec_configs[NEURAL_COLLABORATIVE_FILTERING] = '{
            model_name: "Neural CF",
            embedding_size: 128,
            hidden_layers: 4,
            hidden_units: 256,
            parameters: 20000000,
            flops_per_sample: 60000,
            benchmark_score: 0.82,  // AUC score
            benchmark_metric: "AUC"
        };
    endfunction
    
    // Configure recommendation dataset
    virtual function void configure_rec_dataset();
        case (rec_dataset)
            "MovieLens": begin
                num_users = 138493;
                num_items = 27278;
                config.num_samples = 20000263;  // MovieLens 20M ratings
                config.dataset_name = "MovieLens 20M";
                config.target_accuracy = 0.85;  // AUC
            end
            "Amazon": begin
                num_users = 1000000;
                num_items = 500000;
                config.num_samples = 50000000;
                config.dataset_name = "Amazon Product Reviews";
                config.target_accuracy = 0.82;
            end
            "Netflix": begin
                num_users = 480189;
                num_items = 17770;
                config.num_samples = 100480507;
                config.dataset_name = "Netflix Prize";
                config.target_accuracy = 0.88;
            end
            "Yelp": begin
                num_users = 1968703;
                num_items = 209393;
                config.num_samples = 8021122;
                config.dataset_name = "Yelp Dataset";
                config.target_accuracy = 0.80;
            end
            default: begin
                `uvm_warning(get_type_name(), $sformatf("Unknown dataset: %s, using MovieLens", rec_dataset))
                rec_dataset = "MovieLens";
                configure_rec_dataset();
            end
        endcase
    endfunction
    
    // Load recommendation dataset
    virtual function bit load_rec_dataset();
        `uvm_info(get_type_name(), $sformatf("Loading %s recommendation dataset", rec_dataset), UVM_MEDIUM)
        
        // Generate synthetic recommendation data
        generate_synthetic_rec_data();
        
        return 1;
    endfunction
    
    // Generate synthetic recommendation data
    virtual function void generate_synthetic_rec_data();
        // Generate user and item embeddings
        generate_embeddings();
        
        // Generate training interactions
        int num_train = config.num_samples * 8 / 10;  // 80% for training
        int num_test = config.num_samples - num_train;  // 20% for testing
        
        training_interactions = new[num_train];
        test_interactions = new[num_test];
        
        // Generate training data
        for (int i = 0; i < num_train; i++) begin
            training_interactions[i].user_id = $urandom_range(0, num_users - 1);
            training_interactions[i].item_id = $urandom_range(0, num_items - 1);
            training_interactions[i].rating = generate_realistic_rating(
                training_interactions[i].user_id, training_interactions[i].item_id);
            training_interactions[i].predicted_rating = 0.0;  // Will be filled during inference
            training_interactions[i].confidence = 1.0;
        end
        
        // Generate test data
        for (int i = 0; i < num_test; i++) begin
            test_interactions[i].user_id = $urandom_range(0, num_users - 1);
            test_interactions[i].item_id = $urandom_range(0, num_items - 1);
            test_interactions[i].rating = generate_realistic_rating(
                test_interactions[i].user_id, test_interactions[i].item_id);
            test_interactions[i].predicted_rating = 0.0;
            test_interactions[i].confidence = 1.0;
        end
        
        `uvm_info(get_type_name(), $sformatf("Generated synthetic recommendation data: %0d train, %0d test interactions", 
                 num_train, num_test), UVM_MEDIUM)
    endfunction
    
    // Generate user and item embeddings
    virtual function void generate_embeddings();
        // Allocate embedding matrices
        user_embeddings = new[num_users];
        item_embeddings = new[num_items];
        
        // Generate user embeddings
        for (int u = 0; u < num_users; u++) begin
            user_embeddings[u] = new[embedding_dim];
            for (int d = 0; d < embedding_dim; d++) begin
                // Generate embeddings with some structure (not completely random)
                real base_value = $sin(u * 0.01 + d * 0.1);
                real noise = ($urandom_range(0, 1000) / 1000.0 - 0.5) * 0.2;
                user_embeddings[u][d] = base_value + noise;
            end
        end
        
        // Generate item embeddings
        for (int i = 0; i < num_items; i++) begin
            item_embeddings[i] = new[embedding_dim];
            for (int d = 0; d < embedding_dim; d++) begin
                real base_value = $cos(i * 0.01 + d * 0.1);
                real noise = ($urandom_range(0, 1000) / 1000.0 - 0.5) * 0.2;
                item_embeddings[i][d] = base_value + noise;
            end
        end
        
        `uvm_info(get_type_name(), $sformatf("Generated embeddings: %0d users x %0d dims, %0d items x %0d dims", 
                 num_users, embedding_dim, num_items, embedding_dim), UVM_MEDIUM)
    endfunction
    
    // Generate realistic rating based on user-item similarity
    virtual function real generate_realistic_rating(int user_id, int item_id);
        // Calculate dot product similarity between user and item embeddings
        real similarity = 0.0;
        for (int d = 0; d < embedding_dim; d++) begin
            similarity += user_embeddings[user_id][d] * item_embeddings[item_id][d];
        end
        
        // Normalize similarity to rating scale (1-5)
        real normalized_sim = (similarity + 1.0) / 2.0;  // Map [-1,1] to [0,1]
        real rating = 1.0 + normalized_sim * 4.0;  // Map [0,1] to [1,5]
        
        // Add some noise
        real noise = ($urandom_range(0, 1000) / 1000.0 - 0.5) * 0.5;
        rating += noise;
        
        // Clamp to valid range
        if (rating < 1.0) rating = 1.0;
        if (rating > 5.0) rating = 5.0;
        
        return rating;
    endfunction
    
    // Process recommendation batch
    virtual task process_recommendation_batch(int batch_idx);
        time batch_start = $time;
        
        // Simulate feature extraction
        simulate_feature_extraction();
        
        // Simulate model inference
        simulate_rec_inference(batch_idx);
        
        // Simulate ranking and filtering
        simulate_rec_postprocessing();
        
        time batch_end = $time;
        real batch_latency_ms = real'(batch_end - batch_start) / 1e6;
        
        // Update latency
        if (batch_latency_ms > results.latency_ms) begin
            results.latency_ms = batch_latency_ms;
        end
        
        // Update operation count
        if (rec_configs.exists(config.model_type)) begin
            results.total_operations += rec_configs[config.model_type].flops_per_sample * config.batch_size;
        end
    endtask
    
    // Simulate feature extraction
    virtual task simulate_feature_extraction();
        // Feature extraction includes embedding lookup, feature engineering
        time feature_delay = $urandom_range(10, 50) * 1ns * config.batch_size;
        #feature_delay;
    endtask
    
    // Simulate recommendation inference
    virtual task simulate_rec_inference(int batch_idx);
        time inference_delay;
        
        // Model-specific inference timing
        case (config.model_type)
            WIDE_DEEP: begin
                inference_delay = $urandom_range(50, 150) * 1ns;
            end
            DEEP_FM: begin
                inference_delay = $urandom_range(40, 120) * 1ns;
            end
            NEURAL_COLLABORATIVE_FILTERING: begin
                inference_delay = $urandom_range(30, 100) * 1ns;
            end
            default: begin
                inference_delay = $urandom_range(40, 120) * 1ns;
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
        
        // Generate predictions for this batch
        generate_batch_predictions(batch_idx);
    endtask
    
    // Generate predictions for batch
    virtual function void generate_batch_predictions(int batch_idx);
        int start_idx = batch_idx * config.batch_size;
        int end_idx = start_idx + config.batch_size;
        if (end_idx > test_interactions.size()) end_idx = test_interactions.size();
        
        for (int i = start_idx; i < end_idx; i++) begin
            // Simulate prediction based on user-item similarity with some accuracy
            real true_rating = test_interactions[i].rating;
            real noise_level = 1.0 - (config.target_accuracy / 100.0);  // Higher accuracy = less noise
            real noise = ($urandom_range(0, 1000) / 1000.0 - 0.5) * noise_level * 2.0;
            
            test_interactions[i].predicted_rating = true_rating + noise;
            
            // Clamp to valid range
            if (test_interactions[i].predicted_rating < 1.0) test_interactions[i].predicted_rating = 1.0;
            if (test_interactions[i].predicted_rating > 5.0) test_interactions[i].predicted_rating = 5.0;
            
            // Generate confidence score
            real error = $abs(test_interactions[i].predicted_rating - true_rating);
            test_interactions[i].confidence = 1.0 - (error / 4.0);  // Max error is 4 (5-1)
            if (test_interactions[i].confidence < 0.0) test_interactions[i].confidence = 0.0;
        end
    endfunction
    
    // Simulate recommendation post-processing
    virtual task simulate_rec_postprocessing();
        // Post-processing includes ranking, filtering, diversity enhancement
        time postprocess_delay = $urandom_range(20, 100) * 1ns * config.batch_size;
        #postprocess_delay;
    endtask
    
    // Calculate recommendation-specific metrics
    virtual function void calculate_recommendation_metrics();
        calculate_rating_prediction_metrics();
        calculate_ranking_metrics();
        calculate_classification_metrics();
    endfunction
    
    // Calculate rating prediction metrics (RMSE, MAE)
    virtual function void calculate_rating_prediction_metrics();
        real total_squared_error = 0.0;
        real total_absolute_error = 0.0;
        int valid_predictions = 0;
        
        for (int i = 0; i < test_interactions.size(); i++) begin
            if (test_interactions[i].predicted_rating > 0) begin
                real error = test_interactions[i].predicted_rating - test_interactions[i].rating;
                total_squared_error += error * error;
                total_absolute_error += $abs(error);
                valid_predictions++;
            end
        end
        
        if (valid_predictions > 0) begin
            rmse = $sqrt(total_squared_error / real'(valid_predictions));
            mae = total_absolute_error / real'(valid_predictions);
        end
        
        `uvm_info(get_type_name(), $sformatf("Rating Prediction - RMSE: %.3f, MAE: %.3f", rmse, mae), UVM_MEDIUM)
    endfunction
    
    // Calculate ranking metrics (Precision@K, Recall@K, NDCG@K)
    virtual function void calculate_ranking_metrics();
        // Simplified ranking metrics calculation
        // In real implementation, would need to generate top-K recommendations per user
        
        real total_precision = 0.0;
        real total_recall = 0.0;
        real total_ndcg = 0.0;
        int num_users_evaluated = 0;
        
        // Group interactions by user (simplified)
        for (int u = 0; u < num_users && u < 1000; u++) begin  // Limit to 1000 users for simulation
            // Find interactions for this user
            interaction_t user_interactions[];
            int user_count = 0;
            
            // Collect user interactions
            for (int i = 0; i < test_interactions.size(); i++) begin
                if (test_interactions[i].user_id == u) begin
                    user_count++;
                end
            end
            
            if (user_count >= top_k) begin  // Need enough interactions to evaluate
                user_interactions = new[user_count];
                int idx = 0;
                for (int i = 0; i < test_interactions.size(); i++) begin
                    if (test_interactions[i].user_id == u) begin
                        user_interactions[idx] = test_interactions[i];
                        idx++;
                    end
                end
                
                // Calculate metrics for this user
                real user_precision, user_recall, user_ndcg;
                calculate_user_ranking_metrics(user_interactions, user_precision, user_recall, user_ndcg);
                
                total_precision += user_precision;
                total_recall += user_recall;
                total_ndcg += user_ndcg;
                num_users_evaluated++;
            end
        end
        
        if (num_users_evaluated > 0) begin
            precision_at_k = total_precision / real'(num_users_evaluated) * 100.0;
            recall_at_k = total_recall / real'(num_users_evaluated) * 100.0;
            ndcg_at_k = total_ndcg / real'(num_users_evaluated) * 100.0;
        end
        
        `uvm_info(get_type_name(), $sformatf("Ranking Metrics - P@%0d: %.2f%%, R@%0d: %.2f%%, NDCG@%0d: %.2f%%", 
                 top_k, precision_at_k, top_k, recall_at_k, top_k, ndcg_at_k), UVM_MEDIUM)
    endfunction
    
    // Calculate ranking metrics for single user
    virtual function void calculate_user_ranking_metrics(interaction_t user_interactions[], 
                                                        output real precision, output real recall, output real ndcg);
        // Sort by predicted rating (descending)
        sort_interactions_by_prediction(user_interactions);
        
        // Count relevant items in top-K
        int relevant_in_topk = 0;
        int total_relevant = 0;
        real dcg = 0.0;
        real idcg = 0.0;
        
        // Count total relevant items (rating >= 4.0)
        for (int i = 0; i < user_interactions.size(); i++) begin
            if (user_interactions[i].rating >= 4.0) begin
                total_relevant++;
            end
        end
        
        // Calculate metrics for top-K
        for (int i = 0; i < top_k && i < user_interactions.size(); i++) begin
            if (user_interactions[i].rating >= 4.0) begin
                relevant_in_topk++;
                // DCG calculation
                real gain = user_interactions[i].rating - 1.0;  // Gain = rating - 1
                dcg += gain / $log10(i + 2);  // log base 2 approximated as log10
            end
        end
        
        // Calculate IDCG (ideal DCG)
        for (int i = 0; i < top_k && i < total_relevant; i++) begin
            real ideal_gain = 4.0;  // Assume ideal rating is 5.0, so gain = 4.0
            idcg += ideal_gain / $log10(i + 2);
        end
        
        // Calculate final metrics
        precision = (top_k > 0) ? real'(relevant_in_topk) / real'(top_k) : 0.0;
        recall = (total_relevant > 0) ? real'(relevant_in_topk) / real'(total_relevant) : 0.0;
        ndcg = (idcg > 0) ? dcg / idcg : 0.0;
    endfunction
    
    // Sort interactions by predicted rating (descending)
    virtual function void sort_interactions_by_prediction(ref interaction_t interactions[]);
        // Simple bubble sort (for simulation purposes)
        for (int i = 0; i < interactions.size() - 1; i++) begin
            for (int j = 0; j < interactions.size() - 1 - i; j++) begin
                if (interactions[j].predicted_rating < interactions[j+1].predicted_rating) begin
                    interaction_t temp = interactions[j];
                    interactions[j] = interactions[j+1];
                    interactions[j+1] = temp;
                end
            end
        end
    endfunction
    
    // Calculate classification metrics (AUC)
    virtual function void calculate_classification_metrics();
        // Convert ratings to binary classification (relevant/not relevant)
        // Relevant = rating >= 4.0, Not relevant = rating < 4.0
        
        int true_positives = 0;
        int false_positives = 0;
        int true_negatives = 0;
        int false_negatives = 0;
        
        for (int i = 0; i < test_interactions.size(); i++) begin
            bit actual_relevant = (test_interactions[i].rating >= 4.0);
            bit predicted_relevant = (test_interactions[i].predicted_rating >= 4.0);
            
            if (actual_relevant && predicted_relevant) true_positives++;
            else if (!actual_relevant && predicted_relevant) false_positives++;
            else if (!actual_relevant && !predicted_relevant) true_negatives++;
            else false_negatives++;
        end
        
        // Calculate AUC (simplified)
        real tpr = (true_positives + false_negatives > 0) ? 
                   real'(true_positives) / real'(true_positives + false_negatives) : 0.0;
        real fpr = (false_positives + true_negatives > 0) ? 
                   real'(false_positives) / real'(false_positives + true_negatives) : 0.0;
        
        // Simplified AUC calculation (in reality, would need ROC curve)
        auc_score = (1.0 + tpr - fpr) / 2.0;
        
        // Set overall accuracy to AUC
        results.accuracy_top1 = auc_score * 100.0;
        
        `uvm_info(get_type_name(), $sformatf("Classification - AUC: %.3f, TPR: %.3f, FPR: %.3f", 
                 auc_score, tpr, fpr), UVM_MEDIUM)
    endfunction
    
    // Calculate recommendation-specific performance metrics
    virtual function void calculate_rec_specific_metrics();
        // Recommendations per second
        if (results.total_execution_time > 0) begin
            real seconds = real'(results.total_execution_time) / 1e9;
            real recs_per_second = real'(results.total_samples_processed) / seconds;
            `uvm_info(get_type_name(), $sformatf("Recommendations per second: %.2f", recs_per_second), UVM_MEDIUM)
        end
        
        // Model efficiency metrics
        if (rec_configs.exists(config.model_type)) begin
            rec_model_config_t model_cfg = rec_configs[config.model_type];
            
            // AUC per parameter
            real mparams = real'(model_cfg.parameters) / 1e6;
            if (mparams > 0) begin
                real auc_per_mparam = auc_score / mparams;
                `uvm_info(get_type_name(), $sformatf("AUC per MParam: %.4f", auc_per_mparam), UVM_MEDIUM)
            end
            
            // Precision per GFLOP
            real gflops = real'(model_cfg.flops_per_sample) / 1e9;
            if (gflops > 0) begin
                real precision_per_gflop = precision_at_k / gflops;
                `uvm_info(get_type_name(), $sformatf("Precision@K per GFLOP: %.3f", precision_per_gflop), UVM_MEDIUM)
            end
        end
        
        // Coverage and diversity metrics (simplified)
        real item_coverage = calculate_item_coverage();
        `uvm_info(get_type_name(), $sformatf("Item Coverage: %.2f%%", item_coverage), UVM_MEDIUM)
    endfunction
    
    // Calculate item coverage
    virtual function real calculate_item_coverage();
        // Count unique items in recommendations
        bit item_recommended[];
        item_recommended = new[num_items];
        
        for (int i = 0; i < test_interactions.size(); i++) begin
            if (test_interactions[i].predicted_rating >= 4.0) begin
                item_recommended[test_interactions[i].item_id] = 1;
            end
        end
        
        int covered_items = 0;
        for (int i = 0; i < num_items; i++) begin
            if (item_recommended[i]) covered_items++;
        end
        
        return real'(covered_items) / real'(num_items) * 100.0;
    endfunction
    
    // Override print results to include recommendation metrics
    virtual function void print_results();
        super.print_results();
        
        `uvm_info(get_type_name(), "=== RECOMMENDATION SPECIFIC METRICS ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Task: %s", rec_task), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Dataset: %s", rec_dataset), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Users: %0d, Items: %0d", num_users, num_items), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Embedding Dimension: %0d", embedding_dim), UVM_LOW)
        `uvm_info(get_type_name(), "", UVM_LOW)
        `uvm_info(get_type_name(), "Rating Prediction Metrics:", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  RMSE: %.3f", rmse), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  MAE: %.3f", mae), UVM_LOW)
        `uvm_info(get_type_name(), "", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Ranking Metrics (Top-%0d):", top_k), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Precision@%0d: %.2f%%", top_k, precision_at_k), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  Recall@%0d: %.2f%%", top_k, recall_at_k), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  NDCG@%0d: %.2f%%", top_k, ndcg_at_k), UVM_LOW)
        `uvm_info(get_type_name(), "", UVM_LOW)
        `uvm_info(get_type_name(), "Classification Metrics:", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("  AUC: %.3f", auc_score), UVM_LOW)
    endfunction
    
endclass : recommendation_benchmark

`endif // RECOMMENDATION_BENCHMARK_SV