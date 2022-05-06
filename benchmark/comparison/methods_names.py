import enum


class MethodsNames(enum.Enum):
    ML_LDM = 1
    ELECTRE_I = 2
    TOPSIS = 3

    def __str__(self):
        mapping = {
            MethodsNames.ML_LDM: 'ML-LDM',
            MethodsNames.ELECTRE_I: 'ELECTRE I',
            MethodsNames.TOPSIS: 'TOPSIS',
        }
        return mapping[self]
