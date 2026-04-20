import cv2
import numpy as np

cap = cv2.VideoCapture("traffic.mp4")

ret, old_frame = cap.read()
old_frame = cv2.resize(old_frame, (1280, 720))
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

feature_params = dict(
    maxCorners=400,
    qualityLevel=0.1,
    minDistance=5,
    blockSize=7
)

lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

mask = np.zeros_like(old_frame)
colors = np.random.randint(0, 255, (200, 3))


def get_magnitude(new, old):
    return np.sqrt((new[0] - old[0])**2 + (new[1] - old[1])**2)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1280, 720))

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        old_gray = frame_gray.copy()
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]

    avg_magnitude = np.mean([get_magnitude(n, o) for n, o in zip(good_new, good_old)]) if len(good_new) > 0 else 0

    vehicle_count = 0
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)

        mag = get_magnitude(new, old)

        if mag < 2.5:  # skip stationary points
            continue

        vehicle_count += 1
        mask = cv2.line(mask, (a, b), (c, d), colors[i % 200].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, colors[i % 200].tolist(), -1)

    if vehicle_count < 20:
        traffic_status = "CLEAR"
        color = (0, 255, 0)
    elif vehicle_count < 60:
        traffic_status = "MODERATE"
        color = (0, 255, 255)
    else:
        traffic_status = "DENSE"
        color = (0, 0, 255)

    cv2.putText(frame, f"Moving points: {vehicle_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Avg motion: {avg_magnitude:.2f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(frame, f"Traffic: {traffic_status}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
        mask = np.zeros_like(old_frame)

    output = cv2.add(frame, mask)

    cv2.imshow("Optical Flow - Vehicle Tracking", output)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()