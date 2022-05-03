import enum


class TaskType(enum.Enum):
    NUMERIC_ONLY = 1
    HYBRID_CRISP_LINGUISTIC = 2
    HYBRID_FUZZY_LINGUISTIC = 3
