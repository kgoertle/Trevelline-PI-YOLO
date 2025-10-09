from pathlib import Path
from datetime import datetime, timedelta
import csv
import math

class Aggregator:
    def __init__(self, interval_sec=5, session_sec=10):
        """
        interval_sec: length of interval aggregation in seconds
        session_sec: length of session aggregation in seconds (for testing)
        """
        self.interval_sec = interval_sec
        self.session_sec = session_sec

        # Stores raw frame-level counts: list of tuples (timestamp, M, F, O)
        self.frame_data = []

        # Stores aggregated interval metrics
        self.intervals = []

    def push_frame_data(self, timestamp, males, females, other_counts):
        """
        Push frame-level detection counts into the aggregator
        timestamp: datetime object
        males: int
        females: int
        other_counts: dict of {class_name: count} for other objects
                      Only the following classes are summed for O:
                      Feeder, Main_Perch, Wooden_Perch, Sky_Perch, Nesting_Box
        """
        o_count = sum(other_counts.get(c, 0) for c in ['Feeder', 'Main_Perch', 'Wooden_Perch', 'Sky_Perch', 'Nesting_Box'])
        self.frame_data.append((timestamp, males, females, o_count))

    def _aggregate_intervals(self):
        """
        Internal: Aggregate frame data into completed intervals
        """
        if not self.frame_data:
            return []

        self.intervals = []
        start_time = self.frame_data[0][0]
        interval_end = start_time + timedelta(seconds=self.interval_sec)

        interval_counts = []
        for ts, m, f, o in self.frame_data:
            if ts < interval_end:
                interval_counts.append((m, f, o))
            else:
                # Complete current interval
                if interval_counts:
                    self._save_interval(start_time, interval_end, interval_counts)
                # Advance interval
                start_time = interval_end
                interval_end = start_time + timedelta(seconds=self.interval_sec)
                interval_counts = [(m, f, o)]
        
        # Discard incomplete interval at end
        if interval_counts and (len(interval_counts) >= 1):
            last_ts = self.frame_data[-1][0]
            if last_ts >= interval_end:
                self._save_interval(start_time, interval_end, interval_counts)

        return self.intervals

    def _save_interval(self, start_time, end_time, counts):
        """
        Aggregate counts for an interval and append to intervals list
        """
        total_m = sum(c[0] for c in counts)
        total_f = sum(c[1] for c in counts)
        total_o = sum(c[2] for c in counts)

        # Rate per second
        rate_m = total_m / self.interval_sec
        rate_f = total_f / self.interval_sec
        rate_o = total_o / self.interval_sec

        # M:F ratio
        mf_ratio = rate_m / rate_f if rate_f > 0 else float('inf')

        self.intervals.append({
            'start': start_time,
            'end': end_time,
            'M_count': total_m,
            'F_count': total_f,
            'O_count': total_o,
            'M_rate': rate_m,
            'F_rate': rate_f,
            'O_rate': rate_o,
            'MF_ratio': mf_ratio
        })

    def _aggregate_session(self):
        """
        Aggregate completed intervals into session metrics
        """
        completed_intervals = self.intervals
        if not completed_intervals:
            return []

        session_summary = []
        for cls in ['M', 'F', 'O']:
            counts = [i[f'{cls}_rate'] for i in completed_intervals]
            total_count = sum(i[f'{cls}_count'] for i in completed_intervals)
            mean_rate = sum(counts) / len(counts)
            std_dev = math.sqrt(sum((r - mean_rate) ** 2 for r in counts) / (len(counts) - 1)) if len(counts) > 1 else 0.0

            session_summary.append({
                'Class': cls,
                'Total_Count': total_count,
                'Mean_Rate_per_sec': round(mean_rate, 3),
                'Std_Dev_Rate': round(std_dev, 3)
            })

        return session_summary

    def save_results(self, out_folder):
        """
        Save interval and session CSVs to out_folder
        """
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)

        # First, aggregate intervals
        self._aggregate_intervals()
        interval_file = out_folder / "interval_results.csv"
        with interval_file.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Interval_Start', 'Interval_End', 'Class', 'Count', 'Rate_per_sec', 'MF_Ratio'])
            for interval in self.intervals:
                for cls in ['M', 'F', 'O']:
                    writer.writerow([
                        interval['start'].strftime("%H:%M:%S"),
                        interval['end'].strftime("%H:%M:%S"),
                        cls,
                        interval[f'{cls}_count'],
                        round(interval[f'{cls}_rate'], 3),
                        round(interval['MF_ratio'], 3) if cls in ['M', 'F'] else ''
                    ])

        # Then aggregate session
        session_file = out_folder / "session_summary.csv"
        session_data = self._aggregate_session()
        with session_file.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=session_data[0].keys())
            writer.writeheader()
            writer.writerows(session_data)

        return interval_file, session_file
