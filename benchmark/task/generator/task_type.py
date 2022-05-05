import enum


class TaskType(str, enum.Enum):
    NUMERIC_ONLY = 'NUMERIC_ONLY'
    HYBRID_CRISP_LINGUISTIC = 'HYBRID_CRISP_LINGUISTIC'
    HYBRID_FUZZY_LINGUISTIC = 'HYBRID_FUZZY_LINGUISTIC'
    #
    # @classmethod
    # def from_str(cls, label):
    #     if 'NUMERIC' in label:
    #         return TaskType.NUMERIC_ONLY
    #     if 'HYBRID_CRISP_LINGUISTIC' == label:
    #         return TaskType.HYBRID_CRISP_LINGUISTIC
    #     if 'HYBRID_FUZZY_LINGUISTIC' == label:
    #         return TaskType.HYBRID_FUZZY_LINGUISTIC
