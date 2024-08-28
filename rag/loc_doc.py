import elasticsearch_dsl as dsl


class AttackVecor(dsl.AsyncDocument):
    explain: str = dsl.mapped_field(dsl.Text())
    embedding: list[float] = dsl.mapped_field(dsl.DenseVector())

    class Index:
        name = 'attack_vector'


class RootCause(dsl.AsyncDocument):
    explain: str = dsl.mapped_field(dsl.Text())
    embedding: list[float] = dsl.mapped_field(dsl.DenseVector())

    class Index:
        name = 'root_cause'


class Impact(dsl.AsyncDocument):
    explain: str = dsl.mapped_field(dsl.Text())
    embedding: list[float] = dsl.mapped_field(dsl.DenseVector())

    class Index:
        name = 'impact'


class VulnerabilityType(dsl.AsyncDocument):
    explain: str = dsl.mapped_field(dsl.Text())
    embedding: list[float] = dsl.mapped_field(dsl.DenseVector())

    class Index:
        name = 'vulnerability_type'