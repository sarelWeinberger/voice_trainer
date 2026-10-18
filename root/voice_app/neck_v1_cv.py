"""Deterministic landmark ROIs, rigid compensation and residual motion evidence."""
import math
import cv2
import numpy as np

ROI_NAMES = ('left_lateral', 'right_lateral', 'anterior', 'submental', 'jaw', 'shoulders')


def estimate_head_pitch(points, width, height, scale):
    """Frontal-initialized perspective pose; reject flips and poor fits.

    Image coordinates have downward-positive Y. No cervical angle is measured.
    """
    model = np.array([[0,0,0],[0,330,-65],[-225,-170,-135],
                      [225,-170,-135],[-150,150,-125],[150,150,-125]], float)
    camera = np.array([[width,0,width/2],[0,width,height/2],[0,0,1]],float)
    observed = points[[1,152,33,263,61,291]].astype(float)
    depth = width*450/max(scale,1)
    translation = np.array([[(observed[0,0]-width/2)*depth/width],
                            [(observed[0,1]-height/2)*depth/width], [depth]])
    try:
        ok, rotation, translation = cv2.solvePnP(model, observed, camera, np.zeros(4),
            np.zeros((3,1)), translation, True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            return None
        matrix,_ = cv2.Rodrigues(rotation)
        projected,_ = cv2.projectPoints(model,rotation,translation,camera,np.zeros(4))
        error = float(np.mean(np.linalg.norm(projected[:,0]-observed,axis=1))/scale)
        pitch,yaw,_ = cv2.RQDecomp3x3(matrix)[0]
        depths = (model @ matrix.T + translation.reshape(1,3))[:,2]
        if error > .12 or np.min(depths)<=0 or abs(pitch)>75 or abs(yaw)>75:
            return None
        return float(pitch)
    except cv2.error:
        return None


def fit_rigid(source, target):
    """Least-squares similarity transform; deterministic (no random RANSAC)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    a, b = source.mean(0), target.mean(0)
    x, y = source-a, target-b
    denominator = float((x*x).sum())
    if denominator < 1e-8:
        return None
    u, s, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(2)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    scale = float((s * np.diag(correction)).sum() / denominator)
    linear = scale * rotation
    translation = b - a @ linear
    return np.column_stack((linear.T, translation)).astype(np.float32)


def residual_flow(previous, current, transform):
    """Warp previous into current coordinates before measuring local flow."""
    if transform is None:
        return None
    h, w = current.shape
    warped = cv2.warpAffine(previous, transform, (w, h))
    support = cv2.warpAffine(np.ones_like(previous), transform, (w, h)) > .99
    flow = cv2.calcOpticalFlowFarneback(warped, current, None, .5, 3, 15, 3, 5, 1.2, 0)
    return flow, support


def jaw_residualization(motion, jaw_velocity):
    """Ridge temporal regression with intercept, jaw velocity and absolute velocity.

    Returns residual motion and fraction of variation explained. This is an
    association/control signal, not causal separation of jaw mechanics.
    """
    y, x = np.asarray(motion, float), np.asarray(jaw_velocity, float)
    if len(y) < 8 or np.std(x) < 1e-6:
        return y.tolist(), 0.0
    design = np.column_stack((np.ones(len(x)), x, abs(x)))
    penalty = np.diag([0, 1e-5, 1e-5])
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    prediction = design @ coefficients
    residual = y - prediction
    variance = float(((y-y.mean())**2).sum())
    fraction = float(np.clip(1 - (residual**2).sum()/max(variance, 1e-12), 0, 1))
    return abs(residual).tolist(), fraction


def select_frames(rows, limit=24):
    """Reserve temporal coverage before extrema; ordered, reproducible indices."""
    if not rows or limit<=0:
        return []
    count = min(limit, len(rows))
    if count==1:return [0]
    coverage_count=max(2,count-2)
    targets=np.linspace(rows[0]['t'],rows[-1]['t'],coverage_count)
    chosen={min(range(len(rows)),key=lambda i:(abs(rows[i]['t']-target),i)) for target in targets}
    keys = ('anterior_motion', 'lateral_motion', 'jaw_opening', 'head_motion', 'shoulder_motion',
            'shoulders_head_relative_motion')
    for key in keys:
        rank = sorted(range(len(rows)), key=lambda i: (-rows[i]['features'].get(key, 0), i))
        if len(chosen) < count and rows[rank[0]]['features'].get(key, 0) > 0:
            chosen.add(rank[0])
    for index in range(len(rows)):
        if len(chosen) >= count:
            break
        chosen.add(index)
    return sorted(chosen)[:count]


class CVProcessor:
    def __init__(self):
        import mediapipe as mp
        if not hasattr(mp, 'solutions'):
            raise RuntimeError('MediaPipe legacy solutions required (mediapipe==0.10.14)')
        self.face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                                   min_detection_confidence=.5, min_tracking_confidence=.5)
        self.pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=.5,
                                           min_tracking_confidence=.5)
        self.reset()

    def reset(self):
        self.previous = None
        self.previous_face = None
        self.previous_body = None
        self.previous_shoulders = None
        self.previous_jaw = None
        self.previous_t = None
        if hasattr(self.face, 'reset'):
            self.face.reset()
        if hasattr(self.pose, 'reset'):
            self.pose.reset()

    def close(self):
        self.face.close()
        self.pose.close()

    def process(self, frame, t):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_result, pose_result = self.face.process(rgb), self.pose.process(rgb)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lighting = float(np.mean((gray > 25) & (gray < 235)))
        sharpness = min(1.0, float(cv2.Laplacian(gray, cv2.CV_32F).var()) / 100)
        row = dict(t=float(t), features={}, rois={}, visibility={}, lighting=lighting,
                   resolution_quality=min(1.0, min(w, h)/360), sharpness=sharpness,
                   camera_stability=1.0, landmarks={})
        if not face_result.multi_face_landmarks:
            row['warnings'] = ['face_not_visible']
            self.previous = None
            self.previous_face = None
            self.previous_body = self.previous_shoulders = None
            self.previous_jaw = self.previous_t = None
            return row
        points = np.array([[p.x*w, p.y*h] for p in face_result.multi_face_landmarks[0].landmark], np.float32)
        anchors = points[[33, 133, 362, 263, 1, 168]]
        chin, left_eye, right_eye = points[152], points[33], points[263]
        scale = max(1.0, float(np.linalg.norm(left_eye-right_eye)))
        face_quality = min(1.0, scale/65) * lighting * math.sqrt(sharpness)
        jaw = float(np.linalg.norm(points[13]-points[14])/scale)
        row['visibility']['face'] = face_quality
        row['landmarks']['face_anchors'] = anchors.tolist()
        row['landmarks']['chin'] = chin.tolist()
        row['features'].update(jaw_opening=jaw, head_roll_deg=float(math.degrees(math.atan2(
            float(right_eye[1]-left_eye[1]), abs(float(right_eye[0]-left_eye[0]))+1e-8))),
            head_yaw_proxy=float(abs(points[1,0]-(left_eye[0]+right_eye[0])/2)/scale))
        # Face pitch is approximate perspective pose, not a clinical neck angle.
        pitch = estimate_head_pitch(points,w,h,scale)
        if pitch is not None:
            row['features']['head_pitch_deg'] = pitch
        else:
            row.setdefault('warnings',[]).append('head_pose_unreliable')
        body = None
        tracked_shoulders = None
        shoulder_quality = 0.0
        bottom = min(h-1, chin[1]+scale*1.5)
        center = float(chin[0])
        if pose_result.pose_landmarks:
            pose = pose_result.pose_landmarks.landmark
            coords = np.array([[p.x*w,p.y*h] for p in pose],np.float32)
            shoulders = coords[[11,12]]
            visible = all(0 <= p[0] < w and 0 <= p[1] < h for p in shoulders)
            shoulder_quality = min(pose[11].visibility, pose[12].visibility) * float(visible)
            row['landmarks']['shoulders'] = shoulders.tolist()
            shoulder_mid = shoulders.mean(0)
            if shoulder_quality > .25:
                tracked_shoulders = shoulders
                bottom = min(h-1, float(shoulder_mid[1]))
                row['features']['shoulder_height'] = float((shoulder_mid[1]-chin[1])/scale)
                row['features']['shoulder_asymmetry'] = float(abs(shoulders[0,1]-shoulders[1,1])/scale)
            if shoulder_quality > .25 and min(pose[23].visibility,pose[24].visibility) > .5 and all(0<=p[0]<w and 0<=p[1]<h for p in coords[[23,24]]):
                body = coords[[11,12,23,24]]
                row['features']['shoulder_torso_ratio'] = float(np.linalg.norm(shoulder_mid-coords[[23,24]].mean(0))/scale)
        row['visibility']['shoulders'] = float(shoulder_quality)
        length = max(0, bottom-chin[1])
        # Approximate image-space regions; no anatomical muscle segmentation.
        boxes = {
            'left_lateral': (center-.85*scale, chin[1]+.1*length, center-.2*scale, bottom),
            'right_lateral': (center+.2*scale, chin[1]+.1*length, center+.85*scale, bottom),
            'anterior': (center-.35*scale, chin[1]+.25*length, center+.35*scale, bottom),
            'submental': (center-.55*scale, chin[1]-.08*scale, center+.55*scale, chin[1]+.35*length),
            'jaw': (center-.8*scale, points[13,1], center+.8*scale, chin[1]+.1*scale),
            'shoulders': (center-2*scale, bottom-.25*scale, center+2*scale, bottom+.4*scale),
        }
        yaw_penalty = max(.1, 1-row['features']['head_yaw_proxy'])
        for name, raw in boxes.items():
            x1,y1,x2,y2=raw
            area=max(1,(x2-x1)*(y2-y1))
            box=[max(0,int(x1)),max(0,int(y1)),min(w,int(x2)),min(h,int(y2))]
            retained=max(0,box[2]-box[0])*max(0,box[3]-box[1])/area
            geometry=float(length > .3*scale) if name not in ('jaw','shoulders') else 1.0
            reliability=face_quality*retained*geometry*yaw_penalty
            if name=='shoulders': reliability*=shoulder_quality
            row['rois'][name]=box
            row['visibility'][name]=round(float(np.clip(reliability,0,1)),4)
        if self.previous is not None and self.previous.shape==gray.shape:
            transform=fit_rigid(self.previous_face,anchors)
            compensated=residual_flow(self.previous,gray,transform)
            dt=max(.02,t-self.previous_t)
            displacement=float(np.linalg.norm(anchors.mean(0)-self.previous_face.mean(0))/scale)
            row['features']['head_motion']=displacement/dt
            row['features']['jaw_velocity']=(jaw-self.previous_jaw)/dt
            # Background translation estimates camera/body movement independently
            # from face transform. ROI-free border mask avoids most face/neck pixels.
            mask=np.zeros_like(gray);mask[:, :max(1,w//8)]=255;mask[:, -max(1,w//8):]=255
            corners=cv2.goodFeaturesToTrack(self.previous,80,.02,8,mask=mask)
            camera_motion=0.0
            row['camera_measurement_available']=False
            if corners is not None and len(corners)>=8:
                new,status,_=cv2.calcOpticalFlowPyrLK(self.previous,gray,corners,None)
                valid=status[:,0]==1
                if valid.sum()>=8:
                    delta=(new-corners)[valid,0]
                    camera_motion=float(np.linalg.norm(np.median(delta,axis=0))/scale/dt)
                    row['camera_measurement_available']=True
            row['features']['camera_motion']=camera_motion
            row['camera_stability']=float(1/(1+5*camera_motion)) if row['camera_measurement_available'] else .6
            if compensated:
                flow,support=compensated
                for name,box in row['rois'].items():
                    x1,y1,x2,y2=box
                    patch=flow[y1:y2,x1:x2]
                    valid=support[y1:y2,x1:x2]
                    if patch.size and valid.any() and row['visibility'][name]>.15:
                        magnitude=np.linalg.norm(patch,axis=2)[valid]/scale/dt
                        row['features'][name+'_motion']=float(np.median(magnitude))
                        row['features'][name+'_motion_variance']=float(np.var(magnitude))
                        gradients=np.gradient(patch,axis=(0,1)) if min(patch.shape[:2])>=2 else None
                        if gradients:
                            row['features'][name+'_deformation']=float(np.mean(abs(gradients[0])+abs(gradients[1]))/dt)
            row['features']['anterior_motion']=row['features'].get('anterior_motion',0)+row['features'].get('submental_motion',0)
            row['features']['lateral_motion']=max(row['features'].get('left_lateral_motion',0),row['features'].get('right_lateral_motion',0))
            row['features']['lateral_asymmetry']=abs(row['features'].get('left_lateral_motion',0)-row['features'].get('right_lateral_motion',0))
            if body is not None and self.previous_body is not None:
                # Hip translation control: movement of shoulders relative to hips.
                shoulder_delta=body[:2].mean(0)-self.previous_body[:2].mean(0)
                hip_delta=body[2:].mean(0)-self.previous_body[2:].mean(0)
                row['features']['shoulder_motion']=float(np.linalg.norm(shoulder_delta-hip_delta)/scale/dt)
            elif tracked_shoulders is not None and self.previous_shoulders is not None and transform is not None:
                predicted = cv2.transform(self.previous_shoulders[None],transform)[0]
                row['features']['shoulders_head_relative_motion'] = float(
                    np.linalg.norm(tracked_shoulders-predicted,axis=1).mean()/scale/dt)
                row.setdefault('warnings',[]).append('shoulder_motion_not_torso_referenced')
        self.previous,self.previous_face,self.previous_body=gray,anchors,body
        self.previous_shoulders = tracked_shoulders
        self.previous_jaw,self.previous_t=jaw,t
        return row
