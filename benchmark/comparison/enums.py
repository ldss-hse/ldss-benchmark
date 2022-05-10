import enum


class StatisticsNames(str, enum.Enum):
    KENDALL_TAU = 'KENDALL_TAU'
    SPEARMAN_RHO = 'SPEARMAN_RHO'

    def __str__(self):
        return self.value

    @property
    def coefficient_name(self):
        return 'rho' if self is StatisticsNames.SPEARMAN_RHO else 'tau'
