"""The four selected variants; merging NP and RM is deliberately deferred."""
from zebra.model import ZebraMDM
from puzzle_recurrence.model import PuzzleTrajectoryMDM
from puzzle_recurrence.cursor import PuzzleDataCursor

class PuzzleBaselineMDM(PuzzleDataCursor,ZebraMDM):pass
class PuzzleRecurrentMDM(PuzzleDataCursor,PuzzleTrajectoryMDM):pass

def model_class(config):
    return PuzzleRecurrentMDM if config.mechanisms.tt.enabled else PuzzleBaselineMDM
