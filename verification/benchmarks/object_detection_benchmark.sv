// Object Detection Benchmark
// Comprehensive object detection model benchmarks

`ifndef OBJECT_DETECTION_BENCHMARK_SV
`define OBJECT_DETECTION_BENCHMARK_SV

class object_detection_benchmark extends ai_benchmark_base;
    
    // Object detection specific parameters
    string detection_dataset = "COCO";
    real nms_threshold = 0.5;
    real confidence_threshold = 0.5;
    int max_detections_per_image = 100;
    bit enable_multi_scale_testing = 0;
    
    // Detection metrics
    real map_50 = 0.0;      // mAP at IoU=0.5
    real map_75 = 0.0;      // mAP at IoU=0.75
    real map_50_95 = 0.0;   // mAP averaged over IoU=0.5:0.95
    real map_small = 0.0;   // mAP for small objects
    real map_medium = 0.0;  // mAP for medium objects
    real map_large = 0.0;   // mAP for large objects
    
    // Model configurations for object detection
    typedef struct {
        string model_name;
        int input_height;
        int input_width;
        int channels;
        int num_classes;
        longint parameters;
        longint flops;
        real coco_map;
        real fps_estimate;
    } detection_model_config_t;
    
    detection_model_config_t detection_configs[model_type_e];
    
    // Detection results structure
    typedef struct {
        real x, y, width, height;  // Bounding box
        int class_id;
        real confidence;
    } detection_t;
    
    detection_t detected_objects[][];  // Per image detections
    detection_t ground_truth_objects[][];  // Ground truth annotations
    
    `uvm_object_utils_begin(object_detection_benchmark)
        `uvm_field_string(detection_dataset, UVM_ALL_ON)
        `uvm_field_real(nms_threshold, UVM_ALL_ON)
        `uvm_field_real(confidence_threshold, UVM_ALL_ON)
        `uvm_field_int(max_detections_per_image, UVM_ALL_ON)
        `uvm_field_int(enable_multi_scale_testing, UVM_ALL_ON)
        `uvm_field_real(map_50, UVM_ALL_ON)
        `uvm_field_real(map_75, UVM_ALL_ON)
        `uvm_field_real(map_50_95, UVM_ALL_ON)
    `uvm_object_utils_end
    
    function new(string name = "object_detection_benchmark");
        super.new(name);
        initialize_detection_configs();
    endfunction
    
    virtual function string get_benchmark_name();
        return $sformatf("ObjectDetection-%s-%s", config.model_type.name(), detection_dataset);
    endfunction
    
    virtual function void configure_benchmark(benchmark_config_t cfg);
        config = cfg;
        config.benchmark_type = OBJECT_DETECTION;
        
        // Apply model-specific configuration
        if (detection_configs.exists(config.model_type)) begin
            detection_model_config_t model_cfg = detection_configs[config.model_type];
            config.input_height = model_cfg.input_height;
            config.input_width = model_cfg.input_width;
            config.input_channels = model_cfg.channels;
            config.num_classes = model_cfg.num_classes;
            config.target_accuracy = model_cfg.coco_map;
            config.target_throughput_fps = model_cfg.fps_estimate;
        end else begin
            `uvm_warning(get_type_name(), $sformatf("Unknown detection model: %s, using defaults", config.model_type.name()))
            config.input_height = 416;
            config.input_width = 416;
            config.input_channels = 3;
            config.num_classes = 80;  // COCO classes
            config.target_accuracy = 25.0;  // mAP
        end
        
        // Configure dataset
        configure_detection_dataset();
        
        is_initialized = 1;
    endfunction
    
    virtual function bit initialize_benchmark();
        if (!is_initialized) return 0;
        
        `uvm_info(get_type_name(), $sformatf("Initializing %s detection benchmark on %s", 
                 config.model_type.name(), detection_dataset), UVM_MEDIUM)
        
        // Load detection dataset
        if (!load_detection_dataset()) begin
            `uvm_error(get_type_name(), "Failed to load detection dataset")
            return 0;
        end
        
        return 1;
    endfunction
    
    virtual task run_benchmark();
        `uvm_info(get_type_name(), $sformatf("Running object detection on %0d images", 
                 config.num_samples), UVM_MEDIUM)
        
        // Allocate detection results
        detected_objects = new[config.num_samples];
        
        // Process images
        for (int img = 0; img < config.num_samples; img++) begin
            process_detection_image(img);
            
            // Progress reporting
            if (img % 1000 == 0) begin
                `uvm_info(get_type_name(), $sformatf("Processed %0d/%0d images", img, config.num_samples), UVM_MEDIUM)
            end
        end
        
        results.total_samples_processed = config.num_samples;
    endtask
    
    virtual function void analyze_results();
        calculate_detection_metrics();
        calculate_performance_metrics();
        calculate_detection_specific_metrics();
    endfunction
    
    // Initialize detection model configurations
    virtual function void initialize_detection_configs();
        // YOLO models
        detection_configs[YOLO_V3] = '{
            model_name: "YOLOv3",
            input_height: 416,
            input_width: 416,
            channels: 3,
            num_classes: 80,
            parameters: 62000000,
            flops: 65500000000,
            coco_map: 31.0,
            fps_estimate: 20.0
        };
        
        detection_configs[YOLO_V4] = '{
            model_name: "YOLOv4",
            input_height: 608,
            input_width: 608,
            channels: 3,
            num_classes: 80,
            parameters: 64000000,
            flops: 91800000000,
            coco_map: 43.5,
            fps_estimate: 15.0
        };
        
        detection_configs[YOLO_V5] = '{
            model_name: "YOLOv5s",
            input_height: 640,
            input_width: 640,
            channels: 3,
            num_classes: 80,
            parameters: 7200000,
            flops: 16500000000,
            coco_map: 37.4,
            fps_estimate: 45.0
        };
        
        // SSD models
        detection_configs[SSD_MOBILENET] = '{
            model_name: "SSD MobileNet",
            input_height: 300,
            input_width: 300,
            channels: 3,
            num_classes: 91,  // COCO + background
            parameters: 6800000,
            flops: 2300000000,
            coco_map: 22.0,
            fps_estimate: 60.0
        };
        
        // Faster R-CNN
        detection_configs[FASTER_RCNN] = '{
            model_name: "Faster R-CNN ResNet-50",
            input_height: 800,
            input_width: 1333,
            channels: 3,
            num_classes: 81,  // COCO + background
            parameters: 41500000,
            flops: 134000000000,
            coco_map: 37.4,
            fps_estimate: 7.0
        };
    endfunction
    
    // Configure detection dataset
    virtual function void configure_detection_dataset();
        case (detection_dataset)
            "COCO": begin
                config.num_samples = 5000;  // COCO val2017
                config.dataset_name = "COCO 2017";
                config.num_classes = 80;
            end
            "Pascal VOC": begin
                config.num_samples = 4952;  // VOC 2007 test
                config.dataset_name = "Pascal VOC 2007";
                config.num_classes = 20;
                config.input_height = 416;
                config.input_width = 416;
            end
            "Open Images": begin
                config.num_samples = 125436;  // Open Images V6 validation
                config.dataset_name = "Open Images V6";
                config.num_classes = 600;
            end
            default: begin
                `uvm_warning(get_type_name(), $sformatf("Unknown dataset: %s, using COCO", detection_dataset))
                detection_dataset = "COCO";
                configure_detection_dataset();
            end
        endcase
    endfunction
    
    // Load detection dataset
    virtual function bit load_detection_dataset();
        `uvm_info(get_type_name(), $sformatf("Loading %s detection dataset", detection_dataset), UVM_MEDIUM)
        
        // Generate synthetic detection data
        generate_synthetic_detection_data();
        
        return 1;
    endfunction
    
    // Generate synthetic detection data
    virtual function void generate_synthetic_detection_data();
        // Generate input images
        generate_synthetic_data();
        
        // Generate ground truth annotations
        ground_truth_objects = new[config.num_samples];
        
        for (int img = 0; img < config.num_samples; img++) begin
            // Random number of objects per image (1-10)
            int num_objects = $urandom_range(1, 10);
            ground_truth_objects[img] = new[num_objects];
            
            for (int obj = 0; obj < num_objects; obj++) begin
                // Generate random bounding box
                ground_truth_objects[img][obj].x = $urandom_range(0, config.input_width - 50) / real'(config.input_width);
                ground_truth_objects[img][obj].y = $urandom_range(0, config.input_height - 50) / real'(config.input_height);
                ground_truth_objects[img][obj].width = $urandom_range(20, 200) / real'(config.input_width);
                ground_truth_objects[img][obj].height = $urandom_range(20, 200) / real'(config.input_height);
                
                // Ensure box is within image bounds
                if (ground_truth_objects[img][obj].x + ground_truth_objects[img][obj].width > 1.0) begin
                    ground_truth_objects[img][obj].width = 1.0 - ground_truth_objects[img][obj].x;
                end
                if (ground_truth_objects[img][obj].y + ground_truth_objects[img][obj].height > 1.0) begin
                    ground_truth_objects[img][obj].height = 1.0 - ground_truth_objects[img][obj].y;
                end
                
                // Random class
                ground_truth_objects[img][obj].class_id = $urandom_range(0, config.num_classes - 1);
                ground_truth_objects[img][obj].confidence = 1.0;  // Ground truth has perfect confidence
            end
        end
        
        `uvm_info(get_type_name(), $sformatf("Generated synthetic detection data for %0d images", config.num_samples), UVM_MEDIUM)
    endfunction
    
    // Process single detection image
    virtual task process_detection_image(int img_idx);
        time img_start = $time;
        
        // Simulate preprocessing
        simulate_detection_preprocessing();
        
        // Simulate inference
        simulate_detection_inference(img_idx);
        
        // Simulate postprocessing (NMS, etc.)
        simulate_detection_postprocessing(img_idx);
        
        time img_end = $time;
        real img_latency_ms = real'(img_end - img_start) / 1e6;
        
        // Update latency (track maximum)
        if (img_latency_ms > results.latency_ms) begin
            results.latency_ms = img_latency_ms;
        end
        
        // Update operation count
        if (detection_configs.exists(config.model_type)) begin
            results.total_operations += detection_configs[config.model_type].flops;
        end
    endtask
    
    // Simulate detection preprocessing
    virtual task simulate_detection_preprocessing();
        // Preprocessing includes resize, normalization, padding
        time preprocess_delay = $urandom_range(200, 800) * 1ns;  // 200-800ns
        #preprocess_delay;
    endtask
    
    // Simulate detection inference
    virtual task simulate_detection_inference(int img_idx);
        time inference_delay;
        
        // Model-specific inference timing
        case (config.model_type)
            YOLO_V3: inference_delay = $urandom_range(40000, 60000) * 1ns;  // 40-60us
            YOLO_V4: inference_delay = $urandom_range(60000, 80000) * 1ns;  // 60-80us
            YOLO_V5: inference_delay = $urandom_range(20000, 30000) * 1ns;  // 20-30us
            SSD_MOBILENET: inference_delay = $urandom_range(15000, 25000) * 1ns;  // 15-25us
            FASTER_RCNN: inference_delay = $urandom_range(120000, 180000) * 1ns;  // 120-180us
            default: inference_delay = $urandom_range(30000, 50000) * 1ns;  // 30-50us
        endcase
        
        // Scale by precision
        case (config.precision)
            INT8_QUANT: inference_delay = inference_delay / 4;
            FP16_HALF: inference_delay = inference_delay / 2;
            FP32_SINGLE: /* no change */;
            MIXED_PRECISION: inference_delay = inference_delay * 0.7;
        endcase
        
        #inference_delay;
        
        // Generate synthetic detections
        generate_synthetic_detections(img_idx);
    endtask
    
    // Generate synthetic detection results
    virtual function void generate_synthetic_detections(int img_idx);
        // Generate random number of detections
        int num_detections = $urandom_range(0, max_detections_per_image);
        detected_objects[img_idx] = new[num_detections];
        
        for (int det = 0; det < num_detections; det++) begin
            // Generate detection box
            detected_objects[img_idx][det].x = $urandom_range(0, config.input_width - 20) / real'(config.input_width);
            detected_objects[img_idx][det].y = $urandom_range(0, config.input_height - 20) / real'(config.input_height);
            detected_objects[img_idx][det].width = $urandom_range(10, 150) / real'(config.input_width);
            detected_objects[img_idx][det].height = $urandom_range(10, 150) / real'(config.input_height);
            
            // Ensure box is within bounds
            if (detected_objects[img_idx][det].x + detected_objects[img_idx][det].width > 1.0) begin
                detected_objects[img_idx][det].width = 1.0 - detected_objects[img_idx][det].x;
            end
            if (detected_objects[img_idx][det].y + detected_objects[img_idx][det].height > 1.0) begin
                detected_objects[img_idx][det].height = 1.0 - detected_objects[img_idx][det].y;
            end
            
            // Random class and confidence
            detected_objects[img_idx][det].class_id = $urandom_range(0, config.num_classes - 1);
            detected_objects[img_idx][det].confidence = $urandom_range(confidence_threshold * 100, 100) / 100.0;
        end
    endfunction
    
    // Simulate detection postprocessing
    virtual task simulate_detection_postprocessing(int img_idx);
        // Postprocessing includes NMS, confidence filtering, coordinate conversion
        int num_detections = detected_objects[img_idx].size();
        time postprocess_delay = $urandom_range(100, 500) * 1ns * num_detections;  // Scale with detections
        #postprocess_delay;
        
        // Apply NMS simulation (remove some detections)
        apply_nms_simulation(img_idx);
    endtask
    
    // Simulate Non-Maximum Suppression
    virtual function void apply_nms_simulation(int img_idx);
        int original_count = detected_objects[img_idx].size();
        
        // Simulate NMS by randomly removing some detections
        int nms_removals = $urandom_range(0, original_count / 3);  // Remove up to 1/3
        
        if (nms_removals > 0 && original_count > nms_removals) begin
            detection_t filtered_detections[];
            filtered_detections = new[original_count - nms_removals];
            
            // Keep the first N detections (simplified NMS)
            for (int i = 0; i < filtered_detections.size(); i++) begin
                filtered_detections[i] = detected_objects[img_idx][i];
            end
            
            detected_objects[img_idx] = filtered_detections;
        end
    endfunction
    
    // Calculate detection-specific metrics
    virtual function void calculate_detection_metrics();
        real total_precision = 0.0;
        real total_recall = 0.0;
        int valid_images = 0;
        
        // Calculate per-image metrics and average
        for (int img = 0; img < config.num_samples; img++) begin
            real img_precision, img_recall, img_ap;
            calculate_image_metrics(img, img_precision, img_recall, img_ap);
            
            if (img_precision >= 0 && img_recall >= 0) begin
                total_precision += img_precision;
                total_recall += img_recall;
                valid_images++;
            end
        end
        
        if (valid_images > 0) begin
            real avg_precision = total_precision / valid_images;
            real avg_recall = total_recall / valid_images;
            
            // Simplified mAP calculation (in reality, would need proper AP calculation per class)
            map_50 = avg_precision * 100.0;
            map_75 = avg_precision * 0.8 * 100.0;  // Stricter IoU typically gives lower mAP
            map_50_95 = avg_precision * 0.6 * 100.0;  // Average over multiple IoUs
            
            // Set overall accuracy to mAP@0.5
            results.accuracy_top1 = map_50;
        end
        
        `uvm_info(get_type_name(), $sformatf("Detection Metrics: mAP@0.5=%.2f%%, mAP@0.75=%.2f%%, mAP@0.5:0.95=%.2f%%", 
                 map_50, map_75, map_50_95), UVM_MEDIUM)
    endfunction
    
    // Calculate metrics for single image
    virtual function void calculate_image_metrics(int img_idx, output real precision, output real recall, output real ap);
        int true_positives = 0;
        int false_positives = 0;
        int false_negatives = ground_truth_objects[img_idx].size();
        
        // Match detections to ground truth (simplified IoU matching)
        for (int det = 0; det < detected_objects[img_idx].size(); det++) begin
            bit matched = 0;
            
            for (int gt = 0; gt < ground_truth_objects[img_idx].size(); gt++) begin
                real iou = calculate_iou(detected_objects[img_idx][det], ground_truth_objects[img_idx][gt]);
                
                if (iou > 0.5 && detected_objects[img_idx][det].class_id == ground_truth_objects[img_idx][gt].class_id) begin
                    true_positives++;
                    false_negatives--;
                    matched = 1;
                    break;
                end
            end
            
            if (!matched) begin
                false_positives++;
            end
        end
        
        // Calculate precision and recall
        if (true_positives + false_positives > 0) begin
            precision = real'(true_positives) / real'(true_positives + false_positives);
        end else begin
            precision = 0.0;
        end
        
        if (true_positives + false_negatives > 0) begin
            recall = real'(true_positives) / real'(true_positives + false_negatives);
        end else begin
            recall = 0.0;
        end
        
        // Simplified AP calculation
        ap = (precision + recall) / 2.0;
    endfunction
    
    // Calculate Intersection over Union (IoU)
    virtual function real calculate_iou(detection_t det, detection_t gt);
        // Calculate intersection
        real x1 = (det.x > gt.x) ? det.x : gt.x;
        real y1 = (det.y > gt.y) ? det.y : gt.y;
        real x2 = ((det.x + det.width) < (gt.x + gt.width)) ? (det.x + det.width) : (gt.x + gt.width);
        real y2 = ((det.y + det.height) < (gt.y + gt.height)) ? (det.y + det.height) : (gt.y + gt.height);
        
        if (x2 <= x1 || y2 <= y1) return 0.0;  // No intersection
        
        real intersection = (x2 - x1) * (y2 - y1);
        
        // Calculate union
        real det_area = det.width * det.height;
        real gt_area = gt.width * gt.height;
        real union_area = det_area + gt_area - intersection;
        
        if (union_area <= 0.0) return 0.0;
        
        return intersection / union_area;
    endfunction
    
    // Calculate detection-specific performance metrics
    virtual function void calculate_detection_specific_metrics();
        // Average detections per image
        int total_detections = 0;
        for (int img = 0; img < config.num_samples; img++) begin
            total_detections += detected_objects[img].size();
        end
        real avg_detections = real'(total_detections) / real'(config.num_samples);
        
        `uvm_info(get_type_name(), $sformatf("Average detections per image: %.2f", avg_detections), UVM_MEDIUM)
        
        // Detection efficiency metrics
        if (detection_configs.exists(config.model_type)) begin
            detection_model_config_t model_cfg = detection_configs[config.model_type];
            
            // mAP per GFLOP
            real gflops = real'(model_cfg.flops) / 1e9;
            if (gflops > 0) begin
                real map_per_gflop = map_50 / gflops;
                `uvm_info(get_type_name(), $sformatf("mAP per GFLOP: %.3f", map_per_gflop), UVM_MEDIUM)
            end
            
            // mAP per parameter
            real mparams = real'(model_cfg.parameters) / 1e6;
            if (mparams > 0) begin
                real map_per_mparam = map_50 / mparams;
                `uvm_info(get_type_name(), $sformatf("mAP per MParam: %.3f", map_per_mparam), UVM_MEDIUM)
            end
        end
        
        // Speed-accuracy tradeoff
        if (results.throughput_fps > 0) begin
            real accuracy_fps_product = map_50 * results.throughput_fps;
            `uvm_info(get_type_name(), $sformatf("Accuracy-FPS product: %.2f", accuracy_fps_product), UVM_MEDIUM)
        end
    endfunction
    
    // Override print results to include detection metrics
    virtual function void print_results();
        super.print_results();
        
        `uvm_info(get_type_name(), "=== DETECTION SPECIFIC METRICS ===", UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("mAP@0.5: %.2f%%", map_50), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("mAP@0.75: %.2f%%", map_75), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("mAP@0.5:0.95: %.2f%%", map_50_95), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("NMS Threshold: %.2f", nms_threshold), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Confidence Threshold: %.2f", confidence_threshold), UVM_LOW)
        `uvm_info(get_type_name(), $sformatf("Max Detections per Image: %0d", max_detections_per_image), UVM_LOW)
    endfunction
    
endclass : object_detection_benchmark

`endif // OBJECT_DETECTION_BENCHMARK_SV