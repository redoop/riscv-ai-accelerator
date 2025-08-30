// Benchmark Runner
// Orchestrates execution of multiple AI benchmarks

`ifndef BENCHMARK_RUNNER_SV
`define BENCHMARK_RUNNER_SV

class benchmark_runner extends uvm_component;
    
    // Benchmark registry
    ai_benchmark_base benchmarks[$];
    string benchmark_names[$];
    
    // Runner configuration
    bit run_all_benchmarks = 1;
    string selected_benchmarks[$];
    bit parallel_execution = 0;
    int max_parallel_jobs = 4;
    bit generate_report = 1;
    string report_format = "html";  // html, csv, json
    string output_directory = "./benchmark_results";
    
    // Execution tracking
    int total_benchmarks = 0;
    int completed_benchmarks = 0;
    int failed_benchmarks = 0;
    time total_execution_time = 0;
    
    // Results storage
    benchmark_results_t all_results[];
    string benchmark_summaries[$];
    
    `uvm_component_utils_begin(benchmark_runner)
        `uvm_field_int(run_all_benchmarks, UVM_ALL_ON)
        `uvm_field_queue_string(selected_benchmarks, UVM_ALL_ON)
        `uvm_field_int(parallel_execution, UVM_ALL_ON)
        `uvm_field_int(max_parallel_jobs, UVM_ALL_ON)
        `uvm_field_int(generate_report, UVM_ALL_ON)
        `uvm_field_string(report_format, UVM_ALL_ON)
        `uvm_field_string(output_directory, UVM_ALL_ON)
    `uvm_component_utils_end
    
    function new(string name = "benchmark_runner", uvm_component parent = null);
        super.new(name, parent);
    endfunction
    
    // Build phase - register available benchmarks
    virtual function void build_phase(uvm_phase phase);
        super.build_phase(phase);
        
        `uvm_info(get_type_name(), "Building benchmark runner", UVM_MEDIUM)
        
        // Register all available benchmarks
        register_benchmarks();
        
        // Create output directory
        create_output_directory();
    endfunction
    
    // Register all available benchmark types
    virtual function void register_benchmarks();
        // MLPerf benchmarks
        register_mlperf_benchmarks();
        
        // Image classification benchmarks
        register_image_classification_benchmarks();
        
        // Object detection benchmarks
        register_object_detection_benchmarks();
        
        // NLP benchmarks
        register_nlp_benchmarks();
        
        // Recommendation benchmarks
        register_recommendation_benchmarks();
        
        total_benchmarks = benchmarks.size();
        `uvm_info(get_type_name(), $sformatf("Registered %0d benchmarks", total_benchmarks), UVM_MEDIUM)
    endfunction
    
    // Register MLPerf benchmarks
    virtual function void register_mlperf_benchmarks();
        mlperf_inference_benchmark mlperf_resnet50, mlperf_bert, mlperf_ssd;
        benchmark_config_t cfg;
        
        // MLPerf ResNet-50 Inference
        mlperf_resnet50 = mlperf_inference_benchmark::type_id::create("mlperf_resnet50_inference");
        cfg.model_type = RESNET50;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 1;
        mlperf_resnet50.configure_benchmark(cfg);
        benchmarks.push_back(mlperf_resnet50);
        benchmark_names.push_back("MLPerf-ResNet50-Inference");
        
        // MLPerf BERT Inference
        mlperf_bert = mlperf_inference_benchmark::type_id::create("mlperf_bert_inference");
        cfg.model_type = BERT_BASE;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 1;
        mlperf_bert.configure_benchmark(cfg);
        benchmarks.push_back(mlperf_bert);
        benchmark_names.push_back("MLPerf-BERT-Inference");
        
        // MLPerf SSD-MobileNet Inference
        mlperf_ssd = mlperf_inference_benchmark::type_id::create("mlperf_ssd_inference");
        cfg.model_type = SSD_MOBILENET;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 1;
        mlperf_ssd.configure_benchmark(cfg);
        benchmarks.push_back(mlperf_ssd);
        benchmark_names.push_back("MLPerf-SSD-MobileNet-Inference");
    endfunction
    
    // Register image classification benchmarks
    virtual function void register_image_classification_benchmarks();
        image_classification_benchmark resnet50_fp32, resnet50_int8, mobilenet_fp16, efficientnet_b0;
        benchmark_config_t cfg;
        
        // ResNet-50 FP32
        resnet50_fp32 = image_classification_benchmark::type_id::create("resnet50_fp32");
        cfg.model_type = RESNET50;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 32;
        resnet50_fp32.configure_benchmark(cfg);
        benchmarks.push_back(resnet50_fp32);
        benchmark_names.push_back("ResNet50-FP32-ImageNet");
        
        // ResNet-50 INT8
        resnet50_int8 = image_classification_benchmark::type_id::create("resnet50_int8");
        cfg.model_type = RESNET50;
        cfg.precision = INT8_QUANT;
        cfg.batch_size = 64;
        resnet50_int8.configure_benchmark(cfg);
        benchmarks.push_back(resnet50_int8);
        benchmark_names.push_back("ResNet50-INT8-ImageNet");
        
        // MobileNet-V2 FP16
        mobilenet_fp16 = image_classification_benchmark::type_id::create("mobilenet_fp16");
        cfg.model_type = MOBILENET_V2;
        cfg.precision = FP16_HALF;
        cfg.batch_size = 128;
        mobilenet_fp16.configure_benchmark(cfg);
        benchmarks.push_back(mobilenet_fp16);
        benchmark_names.push_back("MobileNet-V2-FP16-ImageNet");
        
        // EfficientNet-B0
        efficientnet_b0 = image_classification_benchmark::type_id::create("efficientnet_b0");
        cfg.model_type = EFFICIENTNET_B0;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 32;
        efficientnet_b0.configure_benchmark(cfg);
        benchmarks.push_back(efficientnet_b0);
        benchmark_names.push_back("EfficientNet-B0-FP32-ImageNet");
    endfunction
    
    // Register object detection benchmarks
    virtual function void register_object_detection_benchmarks();
        object_detection_benchmark yolo_v5, ssd_mobilenet, faster_rcnn;
        benchmark_config_t cfg;
        
        // YOLOv5
        yolo_v5 = object_detection_benchmark::type_id::create("yolo_v5");
        cfg.model_type = YOLO_V5;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 16;
        yolo_v5.configure_benchmark(cfg);
        benchmarks.push_back(yolo_v5);
        benchmark_names.push_back("YOLOv5-FP32-COCO");
        
        // SSD MobileNet
        ssd_mobilenet = object_detection_benchmark::type_id::create("ssd_mobilenet");
        cfg.model_type = SSD_MOBILENET;
        cfg.precision = FP16_HALF;
        cfg.batch_size = 32;
        ssd_mobilenet.configure_benchmark(cfg);
        benchmarks.push_back(ssd_mobilenet);
        benchmark_names.push_back("SSD-MobileNet-FP16-COCO");
        
        // Faster R-CNN
        faster_rcnn = object_detection_benchmark::type_id::create("faster_rcnn");
        cfg.model_type = FASTER_RCNN;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 8;
        faster_rcnn.configure_benchmark(cfg);
        benchmarks.push_back(faster_rcnn);
        benchmark_names.push_back("Faster-RCNN-FP32-COCO");
    endfunction
    
    // Register NLP benchmarks
    virtual function void register_nlp_benchmarks();
        nlp_benchmark bert_base, bert_large, gpt2_small;
        benchmark_config_t cfg;
        
        // BERT Base
        bert_base = nlp_benchmark::type_id::create("bert_base");
        cfg.model_type = BERT_BASE;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 16;
        bert_base.configure_benchmark(cfg);
        bert_base.nlp_task = "text_classification";
        benchmarks.push_back(bert_base);
        benchmark_names.push_back("BERT-Base-TextClassification");
        
        // BERT Large
        bert_large = nlp_benchmark::type_id::create("bert_large");
        cfg.model_type = BERT_LARGE;
        cfg.precision = FP16_HALF;
        cfg.batch_size = 8;
        bert_large.configure_benchmark(cfg);
        bert_large.nlp_task = "question_answering";
        benchmarks.push_back(bert_large);
        benchmark_names.push_back("BERT-Large-QuestionAnswering");
        
        // GPT-2 Small
        gpt2_small = nlp_benchmark::type_id::create("gpt2_small");
        cfg.model_type = GPT2_SMALL;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 4;
        gpt2_small.configure_benchmark(cfg);
        gpt2_small.nlp_task = "language_modeling";
        benchmarks.push_back(gpt2_small);
        benchmark_names.push_back("GPT2-Small-LanguageModeling");
    endfunction
    
    // Register recommendation benchmarks
    virtual function void register_recommendation_benchmarks();
        recommendation_benchmark wide_deep, deep_fm, neural_cf;
        benchmark_config_t cfg;
        
        // Wide & Deep
        wide_deep = recommendation_benchmark::type_id::create("wide_deep");
        cfg.model_type = WIDE_DEEP;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 1024;
        wide_deep.configure_benchmark(cfg);
        benchmarks.push_back(wide_deep);
        benchmark_names.push_back("Wide-Deep-MovieLens");
        
        // DeepFM
        deep_fm = recommendation_benchmark::type_id::create("deep_fm");
        cfg.model_type = DEEP_FM;
        cfg.precision = FP32_SINGLE;
        cfg.batch_size = 512;
        deep_fm.configure_benchmark(cfg);
        benchmarks.push_back(deep_fm);
        benchmark_names.push_back("DeepFM-MovieLens");
        
        // Neural Collaborative Filtering
        neural_cf = recommendation_benchmark::type_id::create("neural_cf");
        cfg.model_type = NEURAL_COLLABORATIVE_FILTERING;
        cfg.precision = FP16_HALF;
        cfg.batch_size = 256;
        neural_cf.configure_benchmark(cfg);
        benchmarks.push_back(neural_cf);
        benchmark_names.push_back("NeuralCF-MovieLens");
    endfunction
    
    // Create output directory
    virtual function void create_output_directory();
        // In a real implementation, would create actual directory
        `uvm_info(get_type_name(), $sformatf("Output directory: %s", output_directory), UVM_MEDIUM)
    endfunction
    
    // Run phase - execute benchmarks
    virtual task run_phase(uvm_phase phase);
        time start_time, end_time;
        
        phase.raise_objection(this, "Running AI benchmarks");
        
        `uvm_info(get_type_name(), "Starting AI benchmark execution", UVM_LOW)
        start_time = $time;
        
        // Filter benchmarks if specific ones are selected
        if (!run_all_benchmarks && selected_benchmarks.size() > 0) begin
            filter_selected_benchmarks();
        end
        
        // Execute benchmarks
        if (parallel_execution && benchmarks.size() > 1) begin
            run_benchmarks_parallel();
        end else begin
            run_benchmarks_sequential();
        end
        
        end_time = $time;
        total_execution_time = end_time - start_time;
        
        `uvm_info(get_type_name(), $sformatf("Benchmark execution completed in %0t", total_execution_time), UVM_LOW)
        
        phase.drop_objection(this, "Benchmarks completed");
    endtask
    
    // Filter benchmarks based on selection
    virtual function void filter_selected_benchmarks();
        ai_benchmark_base filtered_benchmarks[$];
        string filtered_names[$];
        
        for (int i = 0; i < benchmarks.size(); i++) begin
            foreach (selected_benchmarks[j]) begin
                if (benchmark_names[i] == selected_benchmarks[j]) begin
                    filtered_benchmarks.push_back(benchmarks[i]);
                    filtered_names.push_back(benchmark_names[i]);
                    break;
                end
            end
        end
        
        benchmarks = filtered_benchmarks;
        benchmark_names = filtered_names;
        total_benchmarks = benchmarks.size();
        
        `uvm_info(get_type_name(), $sformatf("Filtered to %0d selected benchmarks", total_benchmarks), UVM_MEDIUM)
    endfunction
    
    // Run benchmarks sequentially
    virtual task run_benchmarks_sequential();
        `uvm_info(get_type_name(), $sformatf("Running %0d benchmarks sequentially", benchmarks.size()), UVM_MEDIUM)
        
        for (int i = 0; i < benchmarks.size(); i++) begin
            run_single_benchmark(benchmarks[i], benchmark_names[i]);
        end
    endtask
    
    // Run benchmarks in parallel
    virtual task run_benchmarks_parallel();
        `uvm_info(get_type_name(), $sformatf("Running %0d benchmarks in parallel (max %0d jobs)", 
                 benchmarks.size(), max_parallel_jobs), UVM_MEDIUM)
        
        // Simple parallel execution using fork-join
        int jobs_running = 0;
        int benchmark_idx = 0;
        
        while (benchmark_idx < benchmarks.size()) begin
            // Start new jobs up to the limit
            while (jobs_running < max_parallel_jobs && benchmark_idx < benchmarks.size()) begin
                automatic int idx = benchmark_idx;
                fork
                    begin
                        run_single_benchmark(benchmarks[idx], benchmark_names[idx]);
                        jobs_running--;
                    end
                join_none
                jobs_running++;
                benchmark_idx++;
            end
            
            // Wait for at least one job to complete
            if (jobs_running >= max_parallel_jobs) begin
                wait(jobs_running < max_parallel_jobs);
            end
        end
        
        // Wait for all remaining jobs to complete
        wait(jobs_running == 0);
    endtask
    
    // Run single benchmark
    virtual task run_single_benchmark(ai_benchmark_base benchmark, string name);
        time bench_start, bench_end;
        
        `uvm_info(get_type_name(), $sformatf("Starting benchmark: %s", name), UVM_MEDIUM)
        bench_start = $time;
        
        // Initialize benchmark
        if (!benchmark.initialize_benchmark()) begin
            `uvm_error(get_type_name(), $sformatf("Failed to initialize benchmark: %s", name))
            failed_benchmarks++;
            return;
        end
        
        // Execute benchmark
        fork
            begin
                benchmark.execute_benchmark();
            end
            begin
                // Timeout protection (10 minutes per benchmark)
                #600s;
                `uvm_warning(get_type_name(), $sformatf("Benchmark timeout: %s", name))
            end
        join_any
        disable fork;
        
        bench_end = $time;
        
        if (benchmark.is_completed) begin
            completed_benchmarks++;
            
            // Store results
            all_results = new[all_results.size() + 1](all_results);
            all_results[all_results.size() - 1] = benchmark.results;
            
            // Store summary
            benchmark_summaries.push_back(benchmark.get_summary_string());
            
            `uvm_info(get_type_name(), $sformatf("Completed benchmark: %s in %0t", name, bench_end - bench_start), UVM_MEDIUM)
        end else begin
            failed_benchmarks++;
            `uvm_error(get_type_name(), $sformatf("Benchmark failed or incomplete: %s", name))
        end
    endtask
    
    // Extract phase - generate reports
    virtual function void extract_phase(uvm_phase phase);
        super.extract_phase(phase);
        
        if (generate_report) begin
            generate_benchmark_reports();
        end
    endfunction
    
    // Generate comprehensive benchmark reports
    virtual function void generate_benchmark_reports();
        `uvm_info(get_type_name(), "Generating benchmark reports", UVM_MEDIUM)
        
        case (report_format)
            "html": generate_html_report();
            "csv": generate_csv_report();
            "json": generate_json_report();
            default: begin
                generate_html_report();
                generate_csv_report();
            end
        endcase
    endfunction
    
    // Generate HTML report
    virtual function void generate_html_report();
        string filename = {output_directory, "/benchmark_report.html"};
        int file_handle;
        
        file_handle = $fopen(filename, "w");
        if (file_handle == 0) begin
            `uvm_error(get_type_name(), $sformatf("Cannot create HTML report: %s", filename))
            return;
        end
        
        // HTML header
        $fwrite(file_handle, "<!DOCTYPE html>\n<html>\n<head>\n");
        $fwrite(file_handle, "<title>RISC-V AI Accelerator Benchmark Report</title>\n");
        $fwrite(file_handle, "<style>\n");
        $fwrite(file_handle, "body { font-family: Arial, sans-serif; margin: 20px; }\n");
        $fwrite(file_handle, "table { border-collapse: collapse; width: 100%%; margin: 20px 0; }\n");
        $fwrite(file_handle, "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }\n");
        $fwrite(file_handle, "th { background-color: #f2f2f2; }\n");
        $fwrite(file_handle, ".pass { color: green; font-weight: bold; }\n");
        $fwrite(file_handle, ".fail { color: red; font-weight: bold; }\n");
        $fwrite(file_handle, ".summary { background-color: #f9f9f9; padding: 15px; border-radius: 5px; }\n");
        $fwrite(file_handle, "</style>\n</head>\n<body>\n");
        
        // Report title and summary
        $fwrite(file_handle, "<h1>RISC-V AI Accelerator Benchmark Report</h1>\n");
        $fwrite(file_handle, "<div class=\"summary\">\n");
        $fwrite(file_handle, "<h2>Executive Summary</h2>\n");
        $fwrite(file_handle, "<p><strong>Total Benchmarks:</strong> %0d</p>\n", total_benchmarks);
        $fwrite(file_handle, "<p><strong>Completed:</strong> <span class=\"pass\">%0d</span></p>\n", completed_benchmarks);
        $fwrite(file_handle, "<p><strong>Failed:</strong> <span class=\"fail\">%0d</span></p>\n", failed_benchmarks);
        $fwrite(file_handle, "<p><strong>Total Execution Time:</strong> %s</p>\n", format_time(total_execution_time));
        $fwrite(file_handle, "</div>\n");
        
        // Detailed results table
        $fwrite(file_handle, "<h2>Detailed Results</h2>\n");
        $fwrite(file_handle, "<table>\n");
        $fwrite(file_handle, "<tr><th>Benchmark</th><th>Accuracy</th><th>Latency (ms)</th><th>Throughput (FPS)</th>");
        $fwrite(file_handle, "<th>Power (W)</th><th>TOPS</th><th>Status</th></tr>\n");
        
        for (int i = 0; i < benchmark_names.size() && i < all_results.size(); i++) begin
            string status_class = (all_results[i].accuracy_top1 >= 70.0) ? "pass" : "fail";
            $fwrite(file_handle, "<tr>\n");
            $fwrite(file_handle, "<td>%s</td>\n", benchmark_names[i]);
            $fwrite(file_handle, "<td>%.2f%%</td>\n", all_results[i].accuracy_top1);
            $fwrite(file_handle, "<td>%.2f</td>\n", all_results[i].latency_ms);
            $fwrite(file_handle, "<td>%.2f</td>\n", all_results[i].throughput_fps);
            $fwrite(file_handle, "<td>%.2f</td>\n", all_results[i].power_consumption_watts);
            $fwrite(file_handle, "<td>%.2f</td>\n", all_results[i].tops_achieved);
            $fwrite(file_handle, "<td class=\"%s\">%s</td>\n", status_class, 
                   (all_results[i].accuracy_top1 >= 70.0) ? "PASS" : "FAIL");
            $fwrite(file_handle, "</tr>\n");
        end
        
        $fwrite(file_handle, "</table>\n");
        
        // HTML footer
        $fwrite(file_handle, "<p><em>Report generated at: %s</em></p>\n", $sformatf("%0t", $time));
        $fwrite(file_handle, "</body>\n</html>\n");
        
        $fclose(file_handle);
        `uvm_info(get_type_name(), $sformatf("HTML report generated: %s", filename), UVM_MEDIUM)
    endfunction
    
    // Generate CSV report
    virtual function void generate_csv_report();
        string filename = {output_directory, "/benchmark_results.csv"};
        int file_handle;
        
        file_handle = $fopen(filename, "w");
        if (file_handle == 0) begin
            `uvm_error(get_type_name(), $sformatf("Cannot create CSV report: %s", filename))
            return;
        end
        
        // CSV header
        $fwrite(file_handle, "Benchmark,Accuracy_Top1,Accuracy_Top5,Latency_ms,Throughput_FPS,");
        $fwrite(file_handle, "Power_W,Energy_mJ,TOPS,Bandwidth_GBps,Samples_Processed,Total_Operations\n");
        
        // CSV data
        for (int i = 0; i < benchmark_names.size() && i < all_results.size(); i++) begin
            $fwrite(file_handle, "%s,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%0d,%0d\n",
                   benchmark_names[i],
                   all_results[i].accuracy_top1,
                   all_results[i].accuracy_top5,
                   all_results[i].latency_ms,
                   all_results[i].throughput_fps,
                   all_results[i].power_consumption_watts,
                   all_results[i].energy_per_inference_mj,
                   all_results[i].tops_achieved,
                   all_results[i].memory_bandwidth_gbps,
                   all_results[i].total_samples_processed,
                   all_results[i].total_operations);
        end
        
        $fclose(file_handle);
        `uvm_info(get_type_name(), $sformatf("CSV report generated: %s", filename), UVM_MEDIUM)
    endfunction
    
    // Generate JSON report
    virtual function void generate_json_report();
        string filename = {output_directory, "/benchmark_results.json"};
        int file_handle;
        
        file_handle = $fopen(filename, "w");
        if (file_handle == 0) begin
            `uvm_error(get_type_name(), $sformatf("Cannot create JSON report: %s", filename))
            return;
        end
        
        // JSON structure
        $fwrite(file_handle, "{\n");
        $fwrite(file_handle, "  \"report_info\": {\n");
        $fwrite(file_handle, "    \"total_benchmarks\": %0d,\n", total_benchmarks);
        $fwrite(file_handle, "    \"completed_benchmarks\": %0d,\n", completed_benchmarks);
        $fwrite(file_handle, "    \"failed_benchmarks\": %0d,\n", failed_benchmarks);
        $fwrite(file_handle, "    \"total_execution_time_ns\": %0d\n", total_execution_time);
        $fwrite(file_handle, "  },\n");
        $fwrite(file_handle, "  \"results\": [\n");
        
        for (int i = 0; i < benchmark_names.size() && i < all_results.size(); i++) begin
            $fwrite(file_handle, "    {\n");
            $fwrite(file_handle, "      \"benchmark\": \"%s\",\n", benchmark_names[i]);
            $fwrite(file_handle, "      \"accuracy_top1\": %.2f,\n", all_results[i].accuracy_top1);
            $fwrite(file_handle, "      \"accuracy_top5\": %.2f,\n", all_results[i].accuracy_top5);
            $fwrite(file_handle, "      \"latency_ms\": %.2f,\n", all_results[i].latency_ms);
            $fwrite(file_handle, "      \"throughput_fps\": %.2f,\n", all_results[i].throughput_fps);
            $fwrite(file_handle, "      \"power_watts\": %.2f,\n", all_results[i].power_consumption_watts);
            $fwrite(file_handle, "      \"tops_achieved\": %.2f,\n", all_results[i].tops_achieved);
            $fwrite(file_handle, "      \"samples_processed\": %0d\n", all_results[i].total_samples_processed);
            $fwrite(file_handle, "    }%s\n", (i < benchmark_names.size() - 1) ? "," : "");
        end
        
        $fwrite(file_handle, "  ]\n");
        $fwrite(file_handle, "}\n");
        
        $fclose(file_handle);
        `uvm_info(get_type_name(), $sformatf("JSON report generated: %s", filename), UVM_MEDIUM)
    endfunction
    
    // Utility function to format time
    virtual function string format_time(time t);
        if (t >= 1s) begin
            return $sformatf("%.2fs", real'(t) / 1e9);
        end else if (t >= 1ms) begin
            return $sformatf("%.2fms", real'(t) / 1e6);
        end else if (t >= 1us) begin
            return $sformatf("%.2fus", real'(t) / 1e3);
        end else begin
            return $sformatf("%0dns", t);
        end
    endfunction
    
    // Report phase
    virtual function void report_phase(uvm_phase phase);
        super.report_phase(phase);
        
        `uvm_info(get_type_name(), "=== BENCHMARK RUNNER SUMMARY ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Benchmarks: %0d", total_benchmarks), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Completed: %0d", completed_benchmarks), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Failed: %0d", failed_benchmarks), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Success Rate: %.1f%%", 
                 real'(completed_benchmarks) / real'(total_benchmarks) * 100.0), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Total Execution Time: %s", format_time(total_execution_time)), UVM_LOW)
        
        if (benchmark_summaries.size() > 0) begin
            `uvm_info(get_type_name(), "=== INDIVIDUAL BENCHMARK SUMMARIES ===", UVM_LOW)
            foreach (benchmark_summaries[i]) begin
                `uvm_info(get_type_name(), benchmark_summaries[i], UVM_LOW)
            end
        end
        
        if (generate_report) begin
            `uvm_info(get_type_name(), $sformatf("Reports generated in: %s", output_directory), UVM_LOW)
        end
    endfunction
    
endclass : benchmark_runner

`endif // BENCHMARK_RUNNER_SV