class MetricsCalculator:

    def __init__(self, sr_class):
        self.sr_class = sr_class

    def get_method_loc(self, sr_method):
        result = 0
        result = sr_method.end_line - sr_method.start_line + 1
        return result