import os
import cv2
import torch
import html
import json
import traceback
from flask import Blueprint, request, jsonify, send_from_directory, current_app
from werkzeug.utils import secure_filename
from collections import Counter, defaultdict
from PIL import Image # Added for converting cv2 frame to PIL Image for Timesformer preprocessing
from ultralytics import YOLO

# Import from our new modules
from config import (
    UPLOAD_FOLDER, INFERRED_FRAMES_OUTPUT_DIR, FRAME_INTERVAL_FOR_INFERENCE,
    SAVE_INFERRED_FRAMES, MIN_CONFIDENCE_THRESHOLD, IDX_TO_LABEL_MAP as CLIP_IDX_TO_LABEL_MAP, 
    DEVICE, NUM_FRAMES_FOR_CAPTIONING, N_REPRESENTATIVE_FRAMES_PER_SEGMENT,
    # Generalized segment detection
    SEGMENT_MAX_GAP_SECONDS_BETWEEN_SIG_EVENTS_IN_CHAIN,
    SEGMENT_MIN_SIG_EVENTS_IN_CHAIN,
    SEGMENT_MAX_NORMAL_FRAMES_PERCENTAGE_IN_SPAN,
    # Timesformer specific
    TIMESFORMER_IDX_TO_LABEL_MAP, TIMESFORMER_INPUT_NUM_FRAMES
)
from models import (
    get_clip_processor, get_custom_clip_classifier_model, get_gemini_model,
    # Timesformer related
    get_timesformer_model, preprocess_frames_for_timesformer,
    load_timesformer_model
)
from utils import allowed_file, cleanup_after_processing, ensure_clean_workspace, store_suspicious_detections

video_bp = Blueprint('video_bp', __name__)

# Initialize YOLO model
yolo_model = YOLO('yolov8n.pt')  # Using YOLOv8 nano model

@video_bp.route('/api/upload', methods=['POST'])
def upload_video():
    print(f"\n--- [API /api/upload START] ---")
    # Ensure workspace is clean before processing
    ensure_clean_workspace()
    
    clip_processor = get_clip_processor()
    custom_clip_classifier = get_custom_clip_classifier_model()
    timesformer_model = get_timesformer_model()

    if not clip_processor or not custom_clip_classifier:
        print(f"Error: CLIP model components not loaded.") # Checkpoint
        return jsonify({"success": False, "message": "CLIP Classification model not loaded. Please check server logs."}), 503
    if not timesformer_model:
        print(f"Error: Timesformer model not loaded.") # Checkpoint
        return jsonify({"success": False, "message": "Timesformer model not loaded. Please check server logs."}), 503
    
    print(f"Models loaded successfully.") # Checkpoint

    if 'video' not in request.files:
        print(f"Error: No video file part in request.") # Checkpoint
        return jsonify({"success": False, "message": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '':
        print(f"Error: No selected video file.") # Checkpoint
        return jsonify({"success": False, "message": "No selected video file"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Video filename: {filename}, Proposed filepath: {filepath}") # Checkpoint
        successful_output_data = None
        raw_frames_cache_for_timesformer = {}
        cap = None # Initialize cap outside try block for finally

        try:
            # Clear previous videos and frames
            clear_previous_files()
            
            file.save(filepath)
            print(f"--- Checkpoint: Video saved to {filepath} ---")
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                print(f"Error: Could not open video file {filepath}") # Checkpoint
                return jsonify({"success": False, "message": f"Could not open video file: {filename}"}), 500

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps == 0 or total_frames == 0:
                print(f"Error: Video {filepath} has 0 FPS or 0 total frames.") # Checkpoint
                cap.release()
                return jsonify({"success": False, "message": f"Video has 0 FPS or 0 frames: {filename}"}), 400

            print(f"--- Checkpoint: Video Info - Total frames: {total_frames}, FPS: {fps:.2f} ---")
            print(f"--- Checkpoint: STAGE 1 START - CLIP frame-by-frame analysis (Interval: {FRAME_INTERVAL_FOR_INFERENCE}) ---")

            frame_processing_results = [] 
            video_output_subdir = ""
            if SAVE_INFERRED_FRAMES:
                video_output_subdir = os.path.join(INFERRED_FRAMES_OUTPUT_DIR, os.path.splitext(filename)[0])
                os.makedirs(video_output_subdir, exist_ok=True)
            
            processed_frames_count = 0
            for frame_idx in range(0, total_frames, FRAME_INTERVAL_FOR_INFERENCE):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = cap.read()
                if not ret: continue

                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                inputs = clip_processor(images=image_rgb, return_tensors="pt")
                pixel_values = inputs.pixel_values.to(DEVICE)
                if pixel_values.ndim == 3: pixel_values = pixel_values.unsqueeze(0)

                with torch.no_grad():
                    clip_logits = custom_clip_classifier(pixel_values)
                    clip_probabilities = torch.softmax(clip_logits, dim=1)
                    clip_max_prob, clip_predicted_idx_tensor = torch.max(clip_probabilities, 1)
                    clip_predicted_class_name = CLIP_IDX_TO_LABEL_MAP.get(clip_predicted_idx_tensor.item(), "Unknown")
                    clip_confidence = clip_max_prob.item()

                timestamp_seconds = frame_idx / fps
                minutes = int(timestamp_seconds // 60)
                seconds = timestamp_seconds % 60
                is_potential_crime_frame_by_clip = clip_predicted_class_name.lower() != 'normal'
                
                if is_potential_crime_frame_by_clip:
                    raw_frames_cache_for_timesformer[frame_idx] = Image.fromarray(image_rgb)

                current_frame_data = {
                    'frame_idx': frame_idx, 'timestamp': f"{minutes:02d}:{seconds:05.2f}",
                    'predicted_class': clip_predicted_class_name,
                    'confidence': clip_confidence,
                    'is_potential_crime_frame': is_potential_crime_frame_by_clip,
                    'saved_frame_path': None
                }
                
                if SAVE_INFERRED_FRAMES and is_potential_crime_frame_by_clip and video_output_subdir:
                    safe_class_name = clip_predicted_class_name.replace(' ', '_').replace('/', '_')
                    output_frame_filename = f"{os.path.splitext(filename)[0]}_frame{frame_idx:05d}_{safe_class_name}_{int(clip_confidence*100)}.jpg"
                    full_saved_path = os.path.join(video_output_subdir, output_frame_filename)
                    try:
                        cv2.imwrite(full_saved_path, frame_bgr) 
                        current_frame_data['saved_frame_path'] = os.path.join(os.path.splitext(filename)[0], output_frame_filename).replace('\\\\', '/')
                    except Exception as e_save:
                        print(f"Error saving frame {output_frame_filename}: {e_save}")
                
                frame_processing_results.append(current_frame_data)
                processed_frames_count += 1
            
            print(f"--- Checkpoint: STAGE 1 END - CLIP analysis complete. Processed {processed_frames_count} frames. Found {len(raw_frames_cache_for_timesformer)} potential crime frames. ---")
            

            if not frame_processing_results:
                print("Error: No frames were processed by CLIP.") # Checkpoint
                return jsonify({"success": False, "message": f"No frames processed for video: {filename}"}), 500

            print(f"--- Checkpoint: STAGE 2 START - Identifying event segments from potential crime frames. ---")
            event_segments = []
            potential_crime_frames_chronological = [
                res for res in frame_processing_results if res.get('is_potential_crime_frame', False)
            ]
            print(f"Total potential crime frames for segment analysis: {len(potential_crime_frames_chronological)}")


            if potential_crime_frames_chronological and fps > 0:
                all_potential_chains = []
                current_chain = []
                for sig_frame_data in potential_crime_frames_chronological:
                    if not current_chain:
                        current_chain.append(sig_frame_data)
                    else:
                        prev_sig_frame_in_chain = current_chain[-1]
                        time_gap_seconds = (sig_frame_data['frame_idx'] - prev_sig_frame_in_chain['frame_idx']) / fps
                        if time_gap_seconds <= SEGMENT_MAX_GAP_SECONDS_BETWEEN_SIG_EVENTS_IN_CHAIN:
                            current_chain.append(sig_frame_data)
                        else:
                            if len(current_chain) >= SEGMENT_MIN_SIG_EVENTS_IN_CHAIN:
                                all_potential_chains.append(list(current_chain))
                            current_chain = [sig_frame_data]
                if len(current_chain) >= SEGMENT_MIN_SIG_EVENTS_IN_CHAIN:
                    all_potential_chains.append(list(current_chain))
                
                print(f"--- Checkpoint: STAGE 2 END - Found {len(all_potential_chains)} potential segment chains. ---")
                print(f"--- Checkpoint: STAGE 3 START - Timesformer analysis for {len(all_potential_chains)} chains. ---")
                
                for i, chain_of_potential_crime_frames in enumerate(all_potential_chains):
                    print(f"  Processing chain {i+1}/{len(all_potential_chains)} with {len(chain_of_potential_crime_frames)} potential crime frames (frames {[f['frame_idx'] for f in chain_of_potential_crime_frames]}) ")
                    
                    span_start_frame_idx = chain_of_potential_crime_frames[0]['frame_idx']
                    span_end_frame_idx = chain_of_potential_crime_frames[-1]['frame_idx']
                    frames_in_full_span = [
                        res for res in frame_processing_results 
                        if res['frame_idx'] >= span_start_frame_idx and res['frame_idx'] <= span_end_frame_idx
                    ]
                    if not frames_in_full_span: 
                        print(f"    Chain {i+1} skipped: No frames in full span (this should not happen).")
                        continue

                    num_potential_crime_frames_in_span = sum(1 for f_span in frames_in_full_span if f_span['is_potential_crime_frame'])
                    num_normal_frames_in_span = len(frames_in_full_span) - num_potential_crime_frames_in_span # Normal according to CLIP flag
                    percentage_normal_in_span = (num_normal_frames_in_span / len(frames_in_full_span)) if frames_in_full_span else 0
                    
                    print(f"    Chain {i+1} span check: {num_normal_frames_in_span} normal frames out of {len(frames_in_full_span)} total in span. Percentage normal: {percentage_normal_in_span*100:.1f}%")
                    if percentage_normal_in_span > SEGMENT_MAX_NORMAL_FRAMES_PERCENTAGE_IN_SPAN:
                        print(f"    Chain {i+1} skipped: normal frame percentage exceeds threshold ({SEGMENT_MAX_NORMAL_FRAMES_PERCENTAGE_IN_SPAN*100:.1f}%).")
                        continue
                    
                    pil_images_for_timesformer = []
                    for frame_data in chain_of_potential_crime_frames:
                        cached_pil_image = raw_frames_cache_for_timesformer.get(frame_data['frame_idx'])
                        if cached_pil_image:
                            pil_images_for_timesformer.append(cached_pil_image)
                        else:
                            print(f"    Warning: Frame {frame_data['frame_idx']} for chain {i+1} not in cache. Timesformer might miss this frame.")
                    
                    if not pil_images_for_timesformer:
                         print(f"    Chain {i+1} skipped: No PIL images retrieved for Timesformer.")
                         continue

                    print(f"    Preparing {len(pil_images_for_timesformer)} frames for Timesformer input (target: {TIMESFORMER_INPUT_NUM_FRAMES})...")
                    timesformer_input_tensor = preprocess_frames_for_timesformer(
                        pil_images_for_timesformer, target_num_frames=TIMESFORMER_INPUT_NUM_FRAMES
                    )

                    segment_final_class_name = "normal" 
                    segment_final_confidence = 0.0

                    # Original input shape from preprocess_frames_for_timesformer is (B, C, T, H, W)
                    # print(f"DEBUG: About to call Timesformer model with input shape: {timesformer_input_tensor.shape}") 
                    
                    # Attempting permutation based on f1.py's alternative: (B, C, T, H, W) -> (B, T, C, H, W)
                    timesformer_input_for_model = timesformer_input_tensor.permute(0, 2, 1, 3, 4)
                    print(f"DEBUG: Permuted input for Timesformer. Original: {timesformer_input_tensor.shape}, Permuted: {timesformer_input_for_model.shape}")

                    with torch.no_grad():
                        # timesformer_logits = timesformer_model(pixel_values=timesformer_input_tensor).logits # Original call
                        timesformer_logits = timesformer_model(pixel_values=timesformer_input_for_model).logits # Call with permuted tensor
                        timesformer_probabilities = torch.softmax(timesformer_logits, dim=1)
                        timesformer_max_prob, timesformer_predicted_idx_tensor = torch.max(timesformer_probabilities, 1)
                        predicted_timesformer_class_name = TIMESFORMER_IDX_TO_LABEL_MAP.get(timesformer_predicted_idx_tensor.item(), "Unknown")
                        predicted_timesformer_confidence = timesformer_max_prob.item()
                    
                    print(f"    Timesformer raw prediction for chain {i+1}: {predicted_timesformer_class_name} (Conf: {predicted_timesformer_confidence:.2f})")

                    if predicted_timesformer_class_name.lower() != 'normal' and predicted_timesformer_confidence >= MIN_CONFIDENCE_THRESHOLD:
                        segment_final_class_name = predicted_timesformer_class_name
                        segment_final_confidence = predicted_timesformer_confidence
                    else:
                        segment_final_class_name = "normal"
                        if predicted_timesformer_class_name.lower() == 'normal':
                             segment_final_confidence = predicted_timesformer_confidence
                        else: 
                             segment_final_confidence = predicted_timesformer_confidence 
                        print(f"    Chain {i+1} classified as '{segment_final_class_name}' with conf {segment_final_confidence:.2f} by Timesformer stage (Reason: low conf or explicit normal pred).")
                    
                    print(f"    Chain {i+1} final classification: {segment_final_class_name}, Confidence: {segment_final_confidence:.2f}")

                    segment_start_frame_data = frames_in_full_span[0]
                    segment_end_frame_data = frames_in_full_span[-1]
                    duration_s = ((segment_end_frame_data['frame_idx'] - segment_start_frame_data['frame_idx']) / fps) + (1/fps)
                    
                    representative_frames_for_segment = []
                    num_potential_frames_in_chain = len(chain_of_potential_crime_frames)
                    if num_potential_frames_in_chain > 0:
                        if num_potential_frames_in_chain <= N_REPRESENTATIVE_FRAMES_PER_SEGMENT:
                            representative_frames_for_segment = chain_of_potential_crime_frames
                        else:
                            indices_to_pick = sorted(list(set([0, num_potential_frames_in_chain - 1] + [
                                int(k * (num_potential_frames_in_chain -1) / (N_REPRESENTATIVE_FRAMES_PER_SEGMENT -1)) 
                                for k in range(1, N_REPRESENTATIVE_FRAMES_PER_SEGMENT -1)])))
                            final_indices = sorted(list(set(indices_to_pick)))[:N_REPRESENTATIVE_FRAMES_PER_SEGMENT]
                            representative_frames_for_segment = [chain_of_potential_crime_frames[k] for k in final_indices if k < num_potential_frames_in_chain]
                            if len(representative_frames_for_segment) < N_REPRESENTATIVE_FRAMES_PER_SEGMENT and len(representative_frames_for_segment) < num_potential_frames_in_chain:
                                step = max(1, num_potential_frames_in_chain // N_REPRESENTATIVE_FRAMES_PER_SEGMENT)
                                temp_frames_from_chain = [chain_of_potential_crime_frames[k] for k in range(0, num_potential_frames_in_chain, step)]
                                combined_repr_frames = {f_data['frame_idx']: f_data for f_data in representative_frames_for_segment}
                                for f_data_chain in temp_frames_from_chain:
                                    if len(combined_repr_frames) < N_REPRESENTATIVE_FRAMES_PER_SEGMENT: combined_repr_frames[f_data_chain['frame_idx']] = f_data_chain
                                    else: break
                                representative_frames_for_segment = list(combined_repr_frames.values())[:N_REPRESENTATIVE_FRAMES_PER_SEGMENT]
                    
                    clip_confidences_in_chain = [round(f['confidence'], 2) for f in chain_of_potential_crime_frames]

                    event_segments.append({
                        'start_frame_idx': segment_start_frame_data['frame_idx'], 'end_frame_idx': segment_end_frame_data['frame_idx'],
                        'start_timestamp': segment_start_frame_data['timestamp'], 'end_timestamp': segment_end_frame_data['timestamp'],
                        'duration_seconds': round(duration_s, 2),
                        'dominant_class': segment_final_class_name, 'confidence': round(segment_final_confidence, 4),
                        'event_count_in_segment': len(chain_of_potential_crime_frames), 
                        'clip_confidences_for_chain_frames': clip_confidences_in_chain,
                        'representative_frames': representative_frames_for_segment,
                        'clip_predictions': [{"frame_idx": res['frame_idx'], "predicted_class": res['predicted_class'], "confidence": res['confidence']} for res in frame_processing_results if res['frame_idx'] >= segment_start_frame_data['frame_idx'] and res['frame_idx'] <= segment_end_frame_data['frame_idx'] and res.get('is_potential_crime_frame', False)],
                        'clip_prediction': chain_of_potential_crime_frames[0]['predicted_class'] if chain_of_potential_crime_frames else "Unknown",
                        'clip_confidence': sum(pred['confidence'] for pred in chain_of_potential_crime_frames) / len(chain_of_potential_crime_frames) if chain_of_potential_crime_frames else 0.0
                    })

                    clip_predictions_for_chain = [{"frame_idx": res['frame_idx'], "predicted_class": res['predicted_class'], "confidence": res['confidence']} for res in frame_processing_results if res['frame_idx'] >= chain_of_potential_crime_frames[0]['frame_idx'] and res['frame_idx'] <= chain_of_potential_crime_frames[-1]['frame_idx'] and res.get('is_potential_crime_frame', False)]
                    print(f"CLIP predictions for chain {i+1}: {clip_predictions_for_chain}")
                print(f"--- Checkpoint: STAGE 3 END - Timesformer analysis complete. Created {len(event_segments)} final segments. ---")
            else:
                print("--- Checkpoint: No potential crime frames for segment analysis OR FPS is 0. Skipping Stages 2 & 3. ---")

            final_non_normal_segment_classes = [
                seg['dominant_class'] for seg in event_segments if seg['dominant_class'].lower() != 'normal'
            ]
            if final_non_normal_segment_classes:
                most_common_overall_prediction = Counter(final_non_normal_segment_classes).most_common(1)[0][0]
            else: 
                most_common_overall_prediction = "normal" 
            
            clip_predictions = [res['predicted_class'] for res in frame_processing_results if res.get('is_potential_crime_frame', False)]
            most_common_clip_prediction = Counter(clip_predictions).most_common(1)[0][0] if clip_predictions else "normal"

            print(f"--- Checkpoint: Final overall most common prediction: {most_common_overall_prediction} ---")
            print(f"--- Checkpoint: Most common CLIP prediction: {most_common_clip_prediction} ---")
            print("Note: For old, unclear, or blurry videos, check Timesformer output. For new, high-quality videos, use CLIP output.")

            successful_output_data = {
                "success": True, "message": "Video processed with two-stage classification.",
                "filename": filename, "total_frames_in_video": total_frames,
                "processed_frames_count": processed_frames_count,
                "most_common_prediction": most_common_overall_prediction,
                "most_common_clip_prediction": most_common_clip_prediction,
                "event_segments": event_segments,
                "clip_predictions": [{"frame_idx": res['frame_idx'], "predicted_class": res['predicted_class'], "confidence": res['confidence']} for res in frame_processing_results if res.get('is_potential_crime_frame', False)]
            }
            print(f"--- Checkpoint: Preparing to return JSON response. ---")

            if successful_output_data:
                # Clean up after successful processing
                cleanup_after_processing(filename)
                return jsonify(successful_output_data)
            else:
                print("Error: successful_output_data not set, returning generic error.")
                return jsonify({"success": False, "message": "An unexpected error occurred during video processing outcome."}), 500
        except Exception as e:
            print(f"--- ERROR during video processing for {filename}: {str(e)} ---")
            traceback.print_exc()
            # Ensure cap is released if it was opened
            if cap and cap.isOpened():
                cap.release()
                print("Video capture released after error.")
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"Cleaned up failed upload: {filepath}")
                except OSError as oe:
                    print(f"Error cleaning up file {filepath}: {oe}")
            return jsonify({"success": False, "message": f"Error processing video: {str(e)}"}), 500
        finally:
            if cap and cap.isOpened():
                cap.release()
                print("Video capture released in finally block.")
            raw_frames_cache_for_timesformer.clear()
            print("Raw frames cache cleared.")
            print("--- [API /api/upload END] ---")

@video_bp.route('/inferred_frames/<path:path_to_file>')
def serve_inferred_frame(path_to_file):
    try:
        # Split the path to get the directory and filename
        path_parts = path_to_file.split('/')
        if len(path_parts) > 1:
            directory = path_parts[0]
            filename = '/'.join(path_parts[1:])
            return send_from_directory(os.path.join(INFERRED_FRAMES_OUTPUT_DIR, directory), filename)
        else:
            return send_from_directory(INFERRED_FRAMES_OUTPUT_DIR, path_to_file)
    except Exception as e:
        print(f"Error serving frame {path_to_file}: {str(e)}")
        return jsonify({"error": "Frame not found"}), 404

@video_bp.route('/api/caption', methods=['POST'])
def caption_video():
    # This route's logic remains unchanged from its previous fully working state.
    # For brevity, I am not re-listing its full content here but assume it's present.
    # It should use get_gemini_model() and process video for captioning.
    # Add print checkpoints if debugging this route specifically.
    print(f"\n--- [API /api/caption START] ---")
    gemini = get_gemini_model()
    if not gemini:
        print("Error: Gemini model not loaded for captioning.")
        return jsonify({"success": False, "message": "Captioning model not loaded. Please check server logs."}), 503

    if 'video' not in request.files:
        return jsonify({"success": False, "message": "No video file part"}), 400
    file = request.files['video']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "message": "No selected file or file type not allowed"}), 400

    filename = secure_filename(file.filename)
    # Create a temporary unique path for captioning to avoid conflicts if same video is uploaded for classification
    caption_temp_filename = f"caption_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, caption_temp_filename)
    cap_caption = None
    try:
        file.save(filepath)
        print(f"Captioning: Video saved temporarily to {filepath}")
        cap_caption = cv2.VideoCapture(filepath)
        if not cap_caption.isOpened():
            return jsonify({"success": False, "message": f"Could not open video: {filename}"}), 500

        total_frames = int(cap_caption.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps_caption = cap_caption.get(cv2.CAP_PROP_FPS) # fps not strictly needed for frame selection here
        if total_frames == 0: # or fps_caption == 0:
            return jsonify({"success": False, "message": "Video has 0 frames."}), 400

        pil_images = []
        indices_to_capture = []
        if total_frames < 1:
            pass 
        elif total_frames < NUM_FRAMES_FOR_CAPTIONING:
            indices_to_capture = list(range(total_frames))
        else:
            raw_indices = torch.linspace(0, total_frames - 1, NUM_FRAMES_FOR_CAPTIONING)
            indices_to_capture = sorted(list(set(int(i) for i in raw_indices)))
        
        print(f"Captioning: Selected {len(indices_to_capture)} frame indices: {indices_to_capture} from {total_frames} total frames.")
        
        for frame_idx_cap in indices_to_capture:
            cap_caption.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_cap)
            ret_cap, frame_cap_bgr = cap_caption.read()
            if ret_cap:
                frame_rgb_cap = cv2.cvtColor(frame_cap_bgr, cv2.COLOR_BGR2RGB)
                pil_img_cap = Image.fromarray(frame_rgb_cap)
                pil_images.append(pil_img_cap)
        
        cap_caption.release() # Release caption video capture early

        if not pil_images:
            return jsonify({"success": False, "message": "Could not extract any frames for captioning."}), 500

        gemini_prompt_parts = ["Analyze these video frames and provide a clear, factual description of the scene and events. Format your response using markdown with the following structure:\n\n## Setting\n[Describe the location and environment]\n\n## Action\n[Describe the sequence of events chronologically]\n\n## Notable Elements\n[Highlight important details, timestamps, and significant observations]\n\nKeep the description objective and concise. Use proper markdown formatting (## for headers, - for bullet points) instead of asterisks."]
        gemini_prompt_parts.extend(pil_images)
        print(f"Captioning: Sending {len(pil_images)} frames to Gemini.")

        response = gemini.generate_content(gemini_prompt_parts)
        gemini_description = response.text
        print(f"Captioning: Gemini Description received.")
        print("--- [API /api/caption END] ---")
        return jsonify({"success": True, "data": {"caption": gemini_description}})

    except Exception as e_cap:
        error_message_cap = html.escape(str(e_cap))
        print(f"CRITICAL Error in caption_video: {error_message_cap}")
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error generating caption: {error_message_cap}"}), 500
    finally:
        if cap_caption is not None and cap_caption.isOpened(): cap_caption.release()
        if os.path.exists(filepath): # Clean up temp caption video
            try: 
                os.remove(filepath)
                print(f"Cleaned up temp caption video: {filepath}")
            except OSError as oe_cap: 
                print(f"Error cleaning up temp caption video {filepath}: {oe_cap}")

@video_bp.route('/api/caption_progress', methods=['GET'])
def caption_progress(): 
    # This route's logic remains unchanged.
    progress_file = os.path.join(UPLOAD_FOLDER, 'caption_progress.json')
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
            return jsonify(progress_data)
        except Exception as e:
            # print(f"Error reading caption progress file: {e}") # Less verbose
            return jsonify({"status": "error", "progress": 0, "message": "Error reading progress."}) 
    return jsonify({"status": "unknown", "progress": 0, "message": "Processing not started or no progress file found"}) 

@video_bp.route('/api/extract-evidence', methods=['POST'])
def extract_evidence():
    print(f"\n--- [API /api/extract-evidence START] ---")
    ensure_clean_workspace()
    
    if 'video' not in request.files:
        print("Error: No video file part")
        return jsonify({'success': False, 'message': 'No video file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        print("Error: No selected video file")
        return jsonify({'success': False, 'message': 'No selected video file'}), 400

    if not allowed_file(file.filename):
        print(f"Error: File type not allowed for {file.filename}")
        return jsonify({'success': False, 'message': 'File type not allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Processing video for object detection: {filepath}")
        
        # Save the uploaded file
        file.save(filepath)
        
        # Create a formatted string representation of the evidence
        evidence_text = []
        evidence_text.append("🔍 Object Detection Results:")
        
        # Read the video
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'success': False, 'message': 'Could not open video file'}), 500

        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Dictionary to store object counts
        object_counts = {}
        
        # Track frame with most detections and frames with suspicious objects
        max_detections = 0
        best_frame = None
        best_frame_idx = None
        best_detections = None
        
        # List to store frames with suspicious objects
        suspicious_frames = []
        suspicious_objects = {'knife', 'gun', 'weapon', 'rifle', 'pistol', 'firearm', 'bottle', 'wine glass', 'beer bottle'}
        MAX_SUSPICIOUS_FRAMES = 5  # Maximum number of frames to display
        frame_sampling_interval = int(fps * 2)  # Sample one frame every 2 seconds
        
        # List to store all suspicious detections with timestamps
        all_suspicious_detections = []
        
        for frame_idx in range(0, total_frames, frame_sampling_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Run YOLO detection
            results = yolo_model(frame)
            detections = results[0].boxes.data.cpu().numpy()
            
            # Process detections
            frame_has_suspicious = False
            suspicious_objects_in_frame = set()
            
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                class_name = yolo_model.names[int(cls)]
                if class_name not in object_counts:
                    object_counts[class_name] = 0
                object_counts[class_name] += 1
                
                # Check for suspicious objects
                if class_name.lower() in suspicious_objects:
                    frame_has_suspicious = True
                    suspicious_objects_in_frame.add(class_name)
            
            # If frame has suspicious objects
            if frame_has_suspicious:
                timestamp = frame_idx / fps
                minutes = int(timestamp // 60)
                seconds = timestamp % 60
                timestamp_str = f"{minutes:02d}:{seconds:05.2f}"
                
                # Store detection for metadata
                all_suspicious_detections.append({
                    'timestamp': timestamp_str,
                    'objects': list(suspicious_objects_in_frame),
                    'frame_idx': frame_idx
                })
                
                # If we haven't reached the display limit, save the frame
                if len(suspicious_frames) < MAX_SUSPICIOUS_FRAMES:
                    suspicious_frames.append({
                        'frame': frame.copy(),
                        'frame_idx': frame_idx,
                        'timestamp': timestamp_str,
                        'detections': detections,
                        'suspicious_objects': list(suspicious_objects_in_frame)
                    })
            
            # Update best frame if this one has more detections
            if len(detections) > max_detections:
                max_detections = len(detections)
                best_frame = frame.copy()
                best_frame_idx = frame_idx
                best_detections = detections
        
        cap.release()
        
        # Store all suspicious detections
        video_name = os.path.splitext(filename)[0]
        store_suspicious_detections(video_name, all_suspicious_detections)
        
        # Format the results
        if object_counts:
            evidence_text.append("\n📊 Detection Summary:")
            for obj_name, count in sorted(object_counts.items()):
                evidence_text.append(f"• {obj_name}: {count} instances detected")
            
            # Add information about suspicious objects if found
            if suspicious_frames:
                evidence_text.append("\n⚠️ Suspicious Objects Timeline:")
                for detection in all_suspicious_detections:
                    evidence_text.append(f"• {', '.join(detection['objects'])} detected at {detection['timestamp']}")
                
                evidence_text.append("\n📸 Selected Frames with Suspicious Objects:")
                for idx, frame_data in enumerate(suspicious_frames):
                    # Draw bounding boxes on the frame
                    frame = frame_data['frame']
                    for det in frame_data['detections']:
                        x1, y1, x2, y2, conf, cls = det
                        class_name = yolo_model.names[int(cls)]
                        if class_name.lower() in suspicious_objects:
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Save the frame
                    output_dir = os.path.join(INFERRED_FRAMES_OUTPUT_DIR, video_name)
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"suspicious_frame_{frame_data['frame_idx']}.jpg")
                    cv2.imwrite(output_path, frame)
                    
                    evidence_text.append(f"• Frame at {frame_data['timestamp']} showing {', '.join(frame_data['suspicious_objects'])}")
                    evidence_text.append(f"  Frame saved as: {os.path.join(video_name, f'suspicious_frame_{frame_data['frame_idx']}.jpg')}")
            
            # Add information about the best frame
            if best_frame is not None:
                timestamp = best_frame_idx / fps
                minutes = int(timestamp // 60)
                seconds = timestamp % 60
                evidence_text.append(f"\n📸 Frame with most detections ({max_detections} objects) at {minutes:02d}:{seconds:05.2f}")
                
                # Draw bounding boxes on the best frame
                for det in best_detections:
                    x1, y1, x2, y2, conf, cls = det
                    class_name = yolo_model.names[int(cls)]
                    cv2.rectangle(best_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(best_frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Save the annotated frame
                output_dir = os.path.join(INFERRED_FRAMES_OUTPUT_DIR, video_name)
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"best_frame_{best_frame_idx}.jpg")
                cv2.imwrite(output_path, best_frame)
                
                # Add the frame path to the response
                evidence_text.append(f"Frame saved as: {os.path.join(video_name, f'best_frame_{best_frame_idx}.jpg')}")
        else:
            evidence_text.append("No objects detected in the video")
        
        # Join all lines with newlines
        formatted_evidence = "\n".join(evidence_text)

        print("Evidence extraction completed successfully")
        print("--- [API /api/extract-evidence END] ---")
        # Clean up after successful processing
        cleanup_after_processing(filename)
        return jsonify({
            'success': True, 
            'data': formatted_evidence
        }), 200
    except Exception as e:
        print(f"Error during evidence extraction: {str(e)}")
        print("--- [API /api/extract-evidence END with ERROR] ---")
        return jsonify({'success': False, 'message': str(e)}), 500
    finally:
        # Clean up the temporary file
        if 'filepath' in locals() and os.path.exists(filepath):
            try:
                os.remove(filepath)
                print(f"Cleaned up temporary video file: {filepath}")
            except OSError as oe:
                print(f"Error cleaning up file {filepath}: {oe}") 

def cleanup_after_processing(filename):
    """Clean up files after processing, but keep the frames"""
    try:
        # Only clean up the video file, not the frames
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(video_path):
            os.unlink(video_path)
    except Exception as e:
        print(f"Error cleaning up video file {filename}: {str(e)}")

def clear_previous_files():
    """Clear all previous videos and frames"""
    try:
        # Clear uploads directory
        for file in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {str(e)}")
        
        # Don't clear inferred frames directory anymore
        # This allows frames to persist for viewing
    except Exception as e:
        print(f"Error clearing previous files: {str(e)}") 

@video_bp.route('/api/classify', methods=['POST'])
def classify_video():
    print(f"\n--- [API /api/classify START] ---")
    ensure_clean_workspace()
    
    if 'video' not in request.files:
        print("Error: No video file part")
        return jsonify({'success': False, 'message': 'No video file part'}), 400
    
    file = request.files['video']
    if file.filename == '':
        print("Error: No selected video file")
        return jsonify({'success': False, 'message': 'No selected video file'}), 400

    if not allowed_file(file.filename):
        print(f"Error: File type not allowed for {file.filename}")
        return jsonify({'success': False, 'message': 'File type not allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        print(f"Processing video for classification: {filepath}")
        
        # Save the uploaded file
        file.save(filepath)
        
        # Get CLIP predictions for the entire video
        clip_predictions = get_clip_predictions(filepath)
        most_common_clip = max(clip_predictions.items(), key=lambda x: x[1])[0]
        
        # Get Timesformer predictions for segments
        event_segments = get_event_segments(filepath)
        
        # Add CLIP predictions to each segment
        for segment in event_segments:
            # Get CLIP prediction for this segment's frames
            segment_frames = get_frames_for_segment(filepath, segment['start_frame_idx'], segment['end_frame_idx'])
            segment_clip_predictions = {}
            
            for frame in segment_frames:
                frame_pred = get_clip_prediction_for_frame(frame)
                if frame_pred:
                    segment_clip_predictions[frame_pred] = segment_clip_predictions.get(frame_pred, 0) + 1
            
            # Get most common CLIP prediction for this segment
            if segment_clip_predictions:
                segment['clip_prediction'] = max(segment_clip_predictions.items(), key=lambda x: x[1])[0]
            else:
                segment['clip_prediction'] = 'Unknown'
        
        # Get overall Timesformer prediction
        most_common_prediction = max(Counter(segment['dominant_class'] for segment in event_segments).items(), key=lambda x: x[1])[0]
        
        # Get representative frames for each segment
        for segment in event_segments:
            segment['representative_frames'] = get_representative_frames(
                filepath, 
                segment['start_frame_idx'], 
                segment['end_frame_idx'],
                segment['dominant_class']
            )
        
        print("Classification completed successfully")
        print("--- [API /api/classify END] ---")
        
        # Clean up after successful processing
        cleanup_after_processing(filename)
        
        return jsonify({
            'success': True,
            'filename': filename,
            'most_common_prediction': most_common_prediction,
            'most_common_clip_prediction': most_common_clip,
            'event_segments': event_segments
        }), 200
        
    except Exception as e:
        print(f"Error during classification: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500

def get_clip_prediction_for_frame(frame):
    """Get CLIP prediction for a single frame"""
    try:
        # Convert frame to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Get CLIP prediction
        prediction = clip_classifier.predict_image(pil_image)
        return prediction
    except Exception as e:
        print(f"Error getting CLIP prediction for frame: {str(e)}")
        return None

def get_frames_for_segment(video_path, start_frame, end_frame):
    """Extract frames for a specific segment"""
    frames = []
    cap = cv2.VideoCapture(video_path)
    
    # Set to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Read frames until end frame
    for _ in range(end_frame - start_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames

def get_clip_predictions(video_path):
    """Get CLIP predictions for the entire video"""
    clip_processor = get_clip_processor()
    custom_clip_classifier = get_custom_clip_classifier_model()
    total_frames = int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
    predictions = {}
    for frame_idx in range(total_frames):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            continue
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = clip_processor(images=image_rgb, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(DEVICE)
        if pixel_values.ndim == 3: pixel_values = pixel_values.unsqueeze(0)
        with torch.no_grad():
            clip_logits = custom_clip_classifier(pixel_values)
            clip_probabilities = torch.softmax(clip_logits, dim=1)
            clip_max_prob, clip_predicted_idx_tensor = torch.max(clip_probabilities, 1)
            clip_predicted_class_name = CLIP_IDX_TO_LABEL_MAP.get(clip_predicted_idx_tensor.item(), "Unknown")
            clip_confidence = clip_max_prob.item()
            predictions[frame_idx] = clip_predicted_class_name
    return predictions

def get_event_segments(video_path):
    """Get event segments from the video with both Timesformer and CLIP predictions"""
    event_segments = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize CLIP models
    clip_processor = get_clip_processor()
    custom_clip_classifier = get_custom_clip_classifier_model()
    
    # Process frames in segments
    current_segment = None
    for frame_idx in range(0, total_frames, FRAME_INTERVAL_FOR_INFERENCE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Get CLIP prediction for this frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = clip_processor(images=frame_rgb, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(DEVICE)
        if pixel_values.ndim == 3:
            pixel_values = pixel_values.unsqueeze(0)
            
        with torch.no_grad():
            clip_logits = custom_clip_classifier(pixel_values)
            clip_probabilities = torch.softmax(clip_logits, dim=1)
            clip_max_prob, clip_predicted_idx_tensor = torch.max(clip_probabilities, 1)
            clip_predicted_class = CLIP_IDX_TO_LABEL_MAP.get(clip_predicted_idx_tensor.item(), "Unknown")
            clip_confidence = float(clip_max_prob.item())  # Ensure it's a Python float
            
        # Get Timesformer prediction (you'll need to implement this part)
        timesformer_prediction = "normal"  # Placeholder
        
        # Create or update segment
        if not current_segment:
            current_segment = {
                'start_frame_idx': frame_idx,
                'start_timestamp': f"{int(frame_idx/fps//60):02d}:{frame_idx/fps%60:05.2f}",
                'end_frame_idx': frame_idx,
                'end_timestamp': f"{int(frame_idx/fps//60):02d}:{frame_idx/fps%60:05.2f}",
                'dominant_class': timesformer_prediction,
                'clip_prediction': clip_predicted_class,
                'clip_confidence': clip_confidence,
                'event_count_in_segment': 1,
                'clip_predictions': [{
                    'frame_idx': frame_idx,
                    'predicted_class': clip_predicted_class,
                    'confidence': clip_confidence
                }]
            }
        else:
            # Update current segment
            current_segment['end_frame_idx'] = frame_idx
            current_segment['end_timestamp'] = f"{int(frame_idx/fps//60):02d}:{frame_idx/fps%60:05.2f}"
            current_segment['event_count_in_segment'] += 1
            current_segment['clip_predictions'].append({
                'frame_idx': frame_idx,
                'predicted_class': clip_predicted_class,
                'confidence': clip_confidence
            })
            
            # Update dominant CLIP prediction
            clip_predictions_counter = Counter(pred['predicted_class'] for pred in current_segment['clip_predictions'])
            if clip_predictions_counter:
                current_segment['clip_prediction'] = clip_predictions_counter.most_common(1)[0][0]
                # Calculate average confidence for the dominant class
                dominant_class = current_segment['clip_prediction']
                dominant_confidences = [pred['confidence'] for pred in current_segment['clip_predictions'] 
                                     if pred['predicted_class'] == dominant_class]
                if dominant_confidences:
                    current_segment['clip_confidence'] = sum(dominant_confidences) / len(dominant_confidences)
                else:
                    current_segment['clip_confidence'] = 0.0
            else:
                current_segment['clip_prediction'] = "Unknown"
                current_segment['clip_confidence'] = 0.0
    
    # Add the last segment if it exists
    if current_segment:
        current_segment['duration_seconds'] = round((current_segment['end_frame_idx'] - current_segment['start_frame_idx']) / fps, 2)
        # Ensure clip_confidence is set
        if 'clip_confidence' not in current_segment:
            current_segment['clip_confidence'] = 0.0
        if 'clip_prediction' not in current_segment:
            current_segment['clip_prediction'] = "Unknown"
        event_segments.append(current_segment)
    
    cap.release()
    return event_segments

def get_representative_frames(video_path, start_frame, end_frame, dominant_class):
    """Get representative frames for a segment"""
    # This function needs to be implemented to return representative frames for a given segment
    # It should return a list of dictionaries, each representing a frame with its index and class
    # For now, we'll return an empty list as a placeholder
    return [] 