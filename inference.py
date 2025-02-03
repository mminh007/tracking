from ultralytics import YOLO
import cv2
import argparse


def main(args):
    """
    """

    cap = cv2.VideoCapture(args.video_path)
    
    w,h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter(*"mp4v")
    out = cv2.VideoWriter(args.output_path, fourcc=fourcc, fps=fps, frameSize=(w,h))

    model = YOLO(args.model_name_or_path)

    frame_counts = 0
    batch_frames = []

    while cap.isOpened():
        
        ret, frame = cap.read()
        if not ret:
            break
        frame_counts += 1
        batch_frames.append(frame)
        
        try:
            if len(batch_frames) == args.batch_frame:
                results = model.track(batch_frames, persist=True,
                                        tracker=args.tracker, iou=args.iou, show=False)

                for frame_idx, result in enumerate(results):
                    annotated_frame = result.plot(font_size=4, line_width=2)

                    out.write(annotated_frame)
            batch_frames = []
        
        except Exception as e:
            print(f"Error when handling frames {frame_counts - len(batch_frames) + 1} to {frame_counts}: {str(e)}")

            batch_frames = []
            continue
        
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--video-path", type=str)
    parser.add_argument("--output-path", str=str)

    parser.add_argument("--batch-frame", type=int, default=2)

    parser.add_argument("--model-name-or-path", type=str, default="yolov8n.pt")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml")

    args = parser.parse_args()

    main(args)