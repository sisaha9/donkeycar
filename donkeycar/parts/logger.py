import logging
from typing import List, Set, Dict, Tuple, Optional


class LoggerPart:
    """
    Log the given values in vehicle memory.
    """
    def __init__(self, inputs: List[str], level: str="INFO", rate: int=1, logger=None):
        self.inputs = inputs
        self.rate = rate
        self.level = logging._nameToLevel.get(level, logging.INFO)
        self.logger = logging.getLogger(logger if logger is not None else "LoggerPart")

        self.values = {}
        self.count = 0
        self.running = True

    def run(self, *args):
        if self.running and args is not None and len(args) == len(self.inputs):
            self.count = (self.count + 1) % (self.rate + 1)
            for i in range(len(self.inputs)):
                field = self.inputs[i]
                value = args[i]
                old_value = self.values.get(field)
                if old_value != value:
                    # always log changes
                    self.logger.log(self.level, f"{field} = {old_value} -> {value}")
                    self.values[field] = value
                elif self.count >= self.rate:
                    self.logger.log(self.level, f"{field} = {value}")

    def shutdown(self):
        self.running = False

class RecordTracker:
        def __init__(self, logger, rec_count_alert, rec_count_alert_cyc, record_alert_color_arr):
            self.last_num_rec_print = 0
            self.dur_alert = 0
            self.force_alert = 0
            self.logger = logger
            self.rec_count_alert = rec_count_alert
            self.rec_count_alert_cyc = rec_count_alert_cyc

        def run(self, num_records):
            if num_records is None:
                return 0

            if self.last_num_rec_print != num_records or self.force_alert:
                self.last_num_rec_print = num_records

                if num_records % 10 == 0:
                    self.logger.info("recorded", num_records, "records")

                if num_records % self.rec_count_alert == 0 or self.force_alert:
                    self.dur_alert = num_records // self.rec_count_alert * self.rec_count_alert_cyc
                    self.force_alert = 0

            if self.dur_alert > 0:
                self.dur_alert -= 1

            if self.dur_alert != 0:
                return self.get_record_alert_color(num_records)

            return 0

        def get_record_alert_color(self, num_records):
            col = (0, 0, 0)
            for count, color in cfg.RECORD_ALERT_COLOR_ARR:
                if num_records >= count:
                    col = color
            return col
