"""Data preparation registry — maps dataset names to preparer modules."""

from . import broad, caruso, euroc, oxiod, repoimu, tumvi

PREPARERS = {
    "broad": broad,
    "euroc": euroc,
    "tumvi": tumvi,
    "oxiod": oxiod,
    "repoimu": repoimu,
    "caruso": caruso,
}
