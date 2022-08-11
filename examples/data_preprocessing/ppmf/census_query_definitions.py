import itertools
from dataclasses import dataclass
from typing import *
from typing_extensions import Literal

from .census_race import CensusRace

@dataclass
class CensusQuery:
    # Alowed ages
    ages: Iterable[int] = None
    # Allowed sexes
    sexes: Iterable[Literal["Male", "Female"]] = None
    # Allowed race IDs
    races: Iterable[CensusRace] = None
    # Allowed values for Hispanic or Latino origin
    HLOs: Iterable[Literal["HLO", "Not HLO"]] = None
    # Whether the query is available at the block level or only the tract level.
    level: Literal["Block", "Tract"] = "Block"
    # Whether or not the table is planned for the 2020 census or not
    in_2020: bool = True


@dataclass(frozen=True, eq=True)
class TableCell:
    tableName: str
    cell: int


_major_race_names = [
    "white",
    "black_or_african_american",
    "american_indian_and_alaska_native",
    "asian",
    "native_hawaiian_and_other_pacific_islander",
    "some_other_race",
]

census_queries = {}  # type: Dict[TableCell, CensusQuery]

################
### Table 1 ###
################

census_queries[TableCell("P1", 1)] = CensusQuery()

################
### Table P6 ###
################

# For each major race, add a query that counts the number of individuals that
# are of that race (alone or in combination with any other races)
for (race_ix, cell) in enumerate(range(2, 8)):
    census_queries[TableCell("P6", cell)] = CensusQuery(
        races=CensusRace.from_predicate(lambda r: r.__dict__[_major_race_names[race_ix]])
    )

################
### Table P7 ###
################

# For each major race, add a query that counts the number of individuals that
# are of Hispanic or latino Origin and that race (alone or in combination with
# any other races)
for (race_ix, cell) in enumerate(range(3, 9)):
    census_queries[TableCell("P7", cell)] = CensusQuery(
        HLOs=["Not HLO"],
        races=CensusRace.from_predicate(lambda r: r.__dict__[_major_race_names[race_ix]]),
    )

# For each major race, add a query that counts the number of individuals that
# are not of Hispanic or latino Origin and that race (alone or in combination
# with any other races)
for (race_ix, cell) in enumerate(range(10, 16)):
    census_queries[TableCell("P7", cell)] = CensusQuery(
        HLOs=["HLO"],
        races=CensusRace.from_predicate(lambda r: r.__dict__[_major_race_names[race_ix]]),
    )

################
### Table P9 ###
################

# P9 Cell 2 is already in P7 Cell 9, so I am skipping it
# P9 Cell 3 is already in P7 Cell 2, so I am skipping it

census_queries[TableCell("P9", 4)] = CensusQuery(
    HLOs=["Not HLO"], races=CensusRace.from_predicate(lambda r: r.num_races() == 1)
)

# For each of the major races, add a query that counts the number of individuals
# that are not of Hispanic or Latino origin and that race alone.
for (race_ix, cell) in enumerate(range(5, 11)):
    census_queries[TableCell("P9", cell)] = CensusQuery(
        HLOs=["Not HLO"], races=[CensusRace(**{_major_race_names[race_ix]: True})]
    )


census_queries[TableCell("P9", 11)] = CensusQuery(
    HLOs=["Not HLO"], races=CensusRace.from_predicate(lambda r: r.num_races() >= 2)
)

census_queries[TableCell("P9", 12)] = CensusQuery(
    HLOs=["Not HLO"], races=CensusRace.from_predicate(lambda r: r.num_races() == 2)
)

# For all pairs of major races, add a query that counts the number of
# individuals that are not of Hispanic or Latino origin and both of those races.
# Note: I checked that the order that races are combined via
# itertools.combinations matches the census table order.
for (races, cell) in zip(itertools.combinations(_major_race_names, 2), range(13, 28)):
    census_queries[TableCell("P9", cell)] = CensusQuery(
        HLOs=["Not HLO"], races=[CensusRace(**{r: True for r in races})]
    )

census_queries[TableCell("P9", 28)] = CensusQuery(
    HLOs=["Not HLO"], races=CensusRace.from_predicate(lambda r: r.num_races() == 3)
)

# For all triples of major races, add a query that counts the number of
# individuals that are not of Hispanic or Latino origin and those races.
for (races, cell) in zip(itertools.combinations(_major_race_names, 3), range(29, 49)):
    census_queries[TableCell("P9", cell)] = CensusQuery(
        HLOs=["Not HLO"], races=[CensusRace(**{r: True for r in races})]
    )

census_queries[TableCell("P9", 49)] = CensusQuery(
    HLOs=["Not HLO"], races=CensusRace.from_predicate(lambda r: r.num_races() == 4)
)

# For all tupes of 4 major races, add a query that counts the number of
# individuals that are not of Hispanic or Latino origin and those races.
for (races, cell) in zip(itertools.combinations(_major_race_names, 4), range(50, 65)):
    census_queries[TableCell("P9", cell)] = CensusQuery(
        HLOs=["Not HLO"], races=[CensusRace(**{r: True for r in races})]
    )

census_queries[TableCell("P9", 65)] = CensusQuery(
    HLOs=["Not HLO"], races=CensusRace.from_predicate(lambda r: r.num_races() == 5)
)

# For all tupes of 5 major races, add a query that counts the number of
# individuals that are not of Hispanic or Latino origin and those races.
for (races, cell) in zip(itertools.combinations(_major_race_names, 5), range(66, 72)):
    census_queries[TableCell("P9", cell)] = CensusQuery(
        HLOs=["Not HLO"], races=[CensusRace(**{r: True for r in races})]
    )

census_queries[TableCell("P9", 72)] = CensusQuery(
    HLOs=["Not HLO"], races=CensusRace.from_predicate(lambda r: r.num_races() == 6)
)

# Note: I'm pretty sure cell 73 is always equal to cell 72. We might consider
# removing one of these queries to avoid giving it double-weight.
census_queries[TableCell("P9", 73)] = CensusQuery(
    HLOs=["Not HLO"],
    races=[
        CensusRace(
            white=True,
            black_or_african_american=True,
            american_indian_and_alaska_native=True,
            asian=True,
            native_hawaiian_and_other_pacific_islander=True,
            some_other_race=True,
        )
    ],
)

#################
### Table P11 ###
#################

age18AndOver = range(18, 116)

census_queries[TableCell("P11", 1)] = CensusQuery(ages=age18AndOver)

census_queries[TableCell("P11", 2)] = CensusQuery(ages=age18AndOver, HLOs=["HLO"])

census_queries[TableCell("P11", 3)] = CensusQuery(ages=age18AndOver, HLOs=["Not HLO"])

census_queries[TableCell("P11", 4)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=CensusRace.from_predicate(lambda r: r.num_races() == 1),
)

# For each major race, add a query that counts the number of individuals 18 and
# older, not of Hispanic or Latino origin, and of that race.
for (race, cell) in zip(_major_race_names, range(5, 11)):
    census_queries[TableCell("P11", cell)] = CensusQuery(
        ages=age18AndOver, HLOs=["Not HLO"], races=[CensusRace(**{race: True})]
    )

census_queries[TableCell("P11", 11)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=CensusRace.from_predicate(lambda r: r.num_races() >= 2),
)

census_queries[TableCell("P11", 12)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=CensusRace.from_predicate(lambda r: r.num_races() == 2),
)

# For each tuple of 2 major races, add a query that counts the number of
# individuals 18 and older, not of Hispanic or Latino origin, and of those
# races.
for (races, cell) in zip(itertools.combinations(_major_race_names, 2), range(13, 28)):
    census_queries[TableCell("P11", cell)] = CensusQuery(
        ages=age18AndOver,
        HLOs=["Not HLO"],
        races=[CensusRace(**{race: True for race in races})],
    )

census_queries[TableCell("P11", 28)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=CensusRace.from_predicate(lambda r: r.num_races() == 3),
)

# For each tuple of 3 major races, add a query that counts the number of
# individuals 18 and older, not of Hispanic or Latino origin, and of those
# races.
for (races, cell) in zip(itertools.combinations(_major_race_names, 3), range(29, 48)):
    census_queries[TableCell("P11", cell)] = CensusQuery(
        ages=age18AndOver,
        HLOs=["Not HLO"],
        races=[CensusRace(**{race: True for race in races})],
    )

census_queries[TableCell("P11", 49)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=CensusRace.from_predicate(lambda r: r.num_races() == 4),
)

# For each tuple of 4 major races, add a query that counts the number of
# individuals 18 and older, not of Hispanic or Latino origin, and of those
# races.
for (races, cell) in zip(itertools.combinations(_major_race_names, 4), range(50, 65)):
    census_queries[TableCell("P11", cell)] = CensusQuery(
        ages=age18AndOver,
        HLOs=["Not HLO"],
        races=[CensusRace(**{race: True for race in races})],
    )

census_queries[TableCell("P11", 65)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=CensusRace.from_predicate(lambda r: r.num_races() == 5),
)

# For each tuple of 5 major races, add a query that counts the number of
# individuals 18 and older, not of Hispanic or Latino origin, and of those
# races.
for (races, cell) in zip(itertools.combinations(_major_race_names, 5), range(66, 72)):
    census_queries[TableCell("P11", cell)] = CensusQuery(
        ages=age18AndOver,
        HLOs=["Not HLO"],
        races=[CensusRace(**{race: True for race in races})],
    )

census_queries[TableCell("P11", 72)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=CensusRace.from_predicate(lambda r: r.num_races() == 6),
)

census_queries[TableCell("P11", 73)] = CensusQuery(
    ages=age18AndOver,
    HLOs=["Not HLO"],
    races=[
        CensusRace(
            white=True,
            black_or_african_american=True,
            american_indian_and_alaska_native=True,
            asian=True,
            native_hawaiian_and_other_pacific_islander=True,
            some_other_race=True,
        )
    ],
)

############################
### Table P12 and P12A-I ###
############################

# P12 Cell 1 is already in P1 Cell 1, so I am omitting it.

_P12_names = [
    "P12",
    "P12A",
    "P12B",
    "P12C",
    "P12D",
    "P12E",
    "P12F",
    "P12G",
    "P12H",
    "P12I",
]

_P12_races = [
    None,
    [CensusRace(white=True)],
    [CensusRace(black_or_african_american=True)],
    [CensusRace(american_indian_and_alaska_native=True)],
    [CensusRace(asian=True)],
    [CensusRace(native_hawaiian_and_other_pacific_islander=True)],
    [CensusRace(some_other_race=True)],
    CensusRace.from_predicate(lambda r: r.num_races() >= 2),
    None,
    [CensusRace(white=True)],
]

_P12_HLOs = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    ["HLO"],
    ["Not HLO"],
]

_age_ranges = [
    range(0, 5),  # under 5 years
    range(5, 10),  # 5 to 9 years
    range(10, 15),  # 10 to 14 years
    range(15, 18),  # 15 to 17 years
    [18, 19],  # 18 and 19 years
    [20],  # 20 years
    [21],  # 21 years
    range(22, 25),  # 22 to 24 years
    range(25, 30),  # 25 to 29 years
    range(30, 35),  # 30 to 34 years
    range(35, 40),  # 35 to 39 years
    range(40, 45),  # 40 to 44 years
    range(45, 50),  # 45 to 49 years
    range(50, 55),  # 50 to 54 years
    range(55, 60),  # 55 to 59 years
    [60, 61],  # 60 and 61 years
    [62, 63, 64],  # 62 to 64 years
    [65, 66],  # 65 and 66 years
    range(67, 70),  # 67 to 69 years
    range(70, 75),  # 70 to 74 years
    range(75, 80),  # 75 to 79 years
    range(80, 85),  # 80 to 85 years
    range(85, 116),  # 85 years and over
]

for (tableName, races, HLOs) in zip(_P12_names, _P12_races, _P12_HLOs):

    census_queries[TableCell(tableName, 1)] = CensusQuery(races=races, HLOs=HLOs)

    census_queries[TableCell(tableName, 2)] = CensusQuery(
        sexes=["Male"], races=races, HLOs=HLOs
    )

    for (ages, cell) in zip(_age_ranges, range(3, 26)):
        census_queries[TableCell(tableName, cell)] = CensusQuery(
            sexes=["Male"], ages=ages, races=races, HLOs=HLOs
        )

    census_queries[TableCell(tableName, 26)] = CensusQuery(
        sexes=["Female"], races=races, HLOs=HLOs
    )

    for (ages, cell) in zip(_age_ranges, range(3, 26)):
        census_queries[TableCell(tableName, cell)] = CensusQuery(
            sexes=["Male"], ages=ages, races=races, HLOs=HLOs
        )

########################################
### Table PCT1 (2020) / PCT12 (2010) ###
########################################

census_queries[TableCell("PCT1", 1)] = CensusQuery(level="Tract")

census_queries[TableCell("PCT1", 2)] = CensusQuery(sexes=["Male"], level="Tract")

for (age, cell) in zip(range(0, 100), range(3, 103)):
    census_queries[TableCell("PCT1", cell)] = CensusQuery(
        sexes=["Male"], ages=[age], level="Tract"
    )

census_queries[TableCell("PCT1", 103)] = CensusQuery(
    sexes=["Male"], ages=range(100, 105), level="Tract"
)

census_queries[TableCell("PCT1", 104)] = CensusQuery(
    sexes=["Male"], ages=range(105, 110), level="Tract"
)

census_queries[TableCell("PCT1", 105)] = CensusQuery(
    sexes=["Male"], ages=range(110, 116), level="Tract"
)

census_queries[TableCell("PCT1", 106)] = CensusQuery(sexes=["Female"], level="Tract")

for (age, cell) in zip(range(0, 100), range(107, 207)):
    census_queries[TableCell("PCT1", cell)] = CensusQuery(
        sexes=["Female"], ages=[age], level="Tract"
    )

census_queries[TableCell("PCT1", 207)] = CensusQuery(
    sexes=["Female"], ages=range(100, 105), level="Tract"
)

census_queries[TableCell("PCT1", 208)] = CensusQuery(
    sexes=["Female"], ages=range(105, 110), level="Tract"
)

census_queries[TableCell("PCT1", 209)] = CensusQuery(
    sexes=["Female"], ages=range(110, 116), level="Tract"
)

######################################
### Tables PCT12A-N (only in 2010) ###
######################################

_PCT12_names = [
    "PCT12A",
    "PCT12B",
    "PCT12C",
    "PCT12D",
    "PCT12E",
    "PCT12F",
    "PCT12G",
    "PCT12H",
    "PCT12I",
    "PCT12J",
    "PCT12K",
    "PCT12L",
    "PCT12M",
    "PCT12N",
]

_PCT12_races = [
    [CensusRace(white=True)],
    [CensusRace(black_or_african_american=True)],
    [CensusRace(american_indian_and_alaska_native=True)],
    [CensusRace(asian=True)],
    [CensusRace(native_hawaiian_and_other_pacific_islander=True)],
    [CensusRace(some_other_race=True)],
    CensusRace.from_predicate(lambda r: r.num_races() >= 2),
    None,
    [CensusRace(white=True)],
    [CensusRace(black_or_african_american=True)],
    [CensusRace(american_indian_and_alaska_native=True)],
    [CensusRace(asian=True)],
    [CensusRace(native_hawaiian_and_other_pacific_islander=True)],
    [CensusRace(some_other_race=True)],
]

_PCT12_HLOs = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
    ["HLO"],
    ["Not HLO"],
    ["Not HLO"],
    ["Not HLO"],
    ["Not HLO"],
    ["Not HLO"],
    ["Not HLO"],
]

for (name, races, HLOs) in zip(_PCT12_names, _PCT12_races, _PCT12_HLOs):
    census_queries[TableCell(name, 1)] = CensusQuery(
        level="Tract", races=races, HLOs=HLOs, in_2020=False
    )

    census_queries[TableCell(name, 2)] = CensusQuery(
        sexes=["Male"], level="Tract", races=races, HLOs=HLOs, in_2020=False
    )

    for (age, cell) in zip(range(0, 100), range(3, 103)):
        census_queries[TableCell(name, cell)] = CensusQuery(
            sexes=["Male"],
            ages=[age],
            level="Tract",
            races=races,
            HLOs=HLOs,
            in_2020=False,
        )

    census_queries[TableCell(name, 103)] = CensusQuery(
        sexes=["Male"],
        ages=range(100, 105),
        level="Tract",
        races=races,
        HLOs=HLOs,
        in_2020=False,
    )

    census_queries[TableCell(name, 104)] = CensusQuery(
        sexes=["Male"],
        ages=range(105, 110),
        level="Tract",
        races=races,
        HLOs=HLOs,
        in_2020=False,
    )

    census_queries[TableCell(name, 105)] = CensusQuery(
        sexes=["Male"],
        ages=range(110, 116),
        level="Tract",
        races=races,
        HLOs=HLOs,
        in_2020=False,
    )

    census_queries[TableCell(name, 106)] = CensusQuery(
        sexes=["Female"], level="Tract", races=races, HLOs=HLOs, in_2020=False
    )

    for (age, cell) in zip(range(0, 100), range(107, 207)):
        census_queries[TableCell(name, cell)] = CensusQuery(
            sexes=["Female"],
            ages=[age],
            level="Tract",
            races=races,
            HLOs=HLOs,
            in_2020=False,
        )

    census_queries[TableCell(name, 207)] = CensusQuery(
        sexes=["Female"],
        ages=range(100, 105),
        level="Tract",
        races=races,
        HLOs=HLOs,
        in_2020=False,
    )

    census_queries[TableCell(name, 208)] = CensusQuery(
        sexes=["Female"],
        ages=range(105, 110),
        level="Tract",
        races=races,
        HLOs=HLOs,
        in_2020=False,
    )

    census_queries[TableCell(name, 209)] = CensusQuery(
        sexes=["Female"],
        ages=range(110, 116),
        level="Tract",
        races=races,
        HLOs=HLOs,
        in_2020=False,
    )


class NHGIS_Mapping:
    def __init__(self):
        self.nhgis2census = {}  # type: Dict[str, str]
        self.census2nhgis = {}  # type: Dict[str, str]

    def add_table_map(self, census: str, nhgis: str):
        self.nhgis2census[nhgis] = census
        self.census2nhgis[census] = nhgis

    def get_nhgis(self, tc: TableCell, type="_dp"):
        return f"{self.census2nhgis[tc.tableName]}{tc.cell:03}{type}"

    def get_census(self, nhgis: str):
        table = nhgis[:-3]
        cell = int(nhgis[-3:])
        return TableCell(self.nhgis2census[table], cell)


nhgis_mapping = NHGIS_Mapping()
nhgis_mapping.add_table_map("P1", "H7V")
nhgis_mapping.add_table_map("P6", "H70")
nhgis_mapping.add_table_map("P7", "H71")
nhgis_mapping.add_table_map("P9", "H73")
nhgis_mapping.add_table_map("P11", "H75")
nhgis_mapping.add_table_map("P12", "H76")
nhgis_mapping.add_table_map("P12A", "H9A")
nhgis_mapping.add_table_map("P12B", "H9B")
nhgis_mapping.add_table_map("P12C", "H9C")
nhgis_mapping.add_table_map("P12D", "H9D")
nhgis_mapping.add_table_map("P12E", "H9E")
nhgis_mapping.add_table_map("P12F", "H9F")
nhgis_mapping.add_table_map("P12G", "H9G")
nhgis_mapping.add_table_map("P12H", "H9H")
nhgis_mapping.add_table_map("P12I", "H9I")

