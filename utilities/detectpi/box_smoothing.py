import numpy as np

class BoxSmoother:
    def __init__(self, max_history=5, alpha=0.6, dist_thresh=None):
        self.max_history = max_history
        self.alpha = alpha
        self.dist_thresh = dist_thresh
        self.history = []

    @staticmethod
    def smooth_angle(prev, new, alpha=0.6):
        """Smooth the angle between two values"""
        diff = ((new - prev + 180) % 360) - 180
        return prev + alpha * diff

    def smooth(self, boxes):
        """
        Smooth the bounding boxes over time based on their distance and class.
        boxes: A list of boxes, each in the form of [cx, cy, w, h, angle, cls].
        """
        smoothed = []
        new_history = []
        for box in boxes:
            cx, cy, w, h, angle, cls = box
            matched = None
            for hx, hy, hw, hh, hangle, hcls in self.history:
                if cls != hcls:
                    continue
                if self.dist_thresh is None or np.linalg.norm([cx - hx, cy - hy]) < self.dist_thresh:
                    matched = (hx, hy, hw, hh, hangle, hcls)
                    break
            if matched:
                hx, hy, hw, hh, hangle, _ = matched
                cx = self.alpha * cx + (1 - self.alpha) * hx
                cy = self.alpha * cy + (1 - self.alpha) * hy
                w = self.alpha * w + (1 - self.alpha) * hw
                h = self.alpha * h + (1 - self.alpha) * hh
                angle = self.smooth_angle(hangle, angle, self.alpha)
                angle = ((angle + 180) % 360) - 180
            smoothed.append([cx, cy, w, h, angle, cls])
            new_history.append([cx, cy, w, h, angle, cls])
        self.history = new_history[-self.max_history:]
        return smoothed
