from __future__ import annotations
from dataclasses import dataclass
from typing import *


@dataclass(frozen=True, eq=True)
class CensusRace:
    """
    Encodes one of the 63 values for the CENRACE column. The 63 different race
    values correspond to all possible non-empty subsets of 6 race categories:
    - White
    - Black or African American
    - American Indian and Alaska Native
    - Asian
    - Native Hawaiian and Other Pacific Islander
    - Some Other Race
    """

    white: bool = False
    black_or_african_american: bool = False
    american_indian_and_alaska_native: bool = False
    asian: bool = False
    native_hawaiian_and_other_pacific_islander: bool = False
    some_other_race: bool = False

    def num_races(self) -> int:
        """`num_races()

        Returns the number of race categories that this race belongs to. Always
        an integer between 1 and 6 inclusive.
        """
        return (
            self.white
            + self.black_or_african_american
            + self.american_indian_and_alaska_native
            + self.asian
            + self.native_hawaiian_and_other_pacific_islander
            + self.some_other_race
        )

    def to_id(self) -> int:
        """`to_id()`
        
        Return the numeric value that encodes this race in the CENRACE column.
        """
        return race_to_id[self]

    def __str__(self):
        return _race_strs[self.to_id()]

    @staticmethod
    def from_id(id: int) -> CensusRace:
        """`from_id(id: int) -> CensusRace
        
        Return the instance of `CensusRace` that is encoded by `id` in the
        CENRACE column.
        """
        return id_to_race[id]

    @staticmethod
    def parse_census_race(race_str: str) -> CensusRace:
        """`parse_census_race(race_str: str) -> CensusRace`

        A helper function that parses the text descriptions of a census race.
        """
        result = CensusRace()
        if race_str.endswith(" alone"):
            race_str = race_str[:-6]
        race_str = race_str.lower()
        races = [r.strip() for r in race_str.split(";") if r != ""]

        for race in races:
            field_name = race.replace(" ", "_")
            if field_name not in result.__dict__.keys():
                raise KeyError(f"Race {race} is not one of the census races")
            result.__dict__[race.replace(" ", "_")] = True
        return result

    @staticmethod
    def from_predicate(pred: Callable[[CensusRace], bool]) -> List[CensusRace]:
        """`from_predicate(pred: Callable[[CensusRace]m, bool]) -> List[CensusRace]`
        
        Given a predicate defined over instances of `CensusRace`, returns the
        list of all `CensusRace` instances that satisfy this predicate.
        """
        return [cr for cr in id_to_race[1:] if pred(cr)]


# Obtained from https://www2.census.gov/programs-surveys/decennial/2020/program-management/data-product-planning/2010-demonstration-data-products/ppmf/2020-05-27-ppmf-record-layout.pdf
_race_strs = [
    "",
    "White alone",
    "Black or African American alone",
    "American Indian and Alaska Native alone",
    "Asian alone",
    "Native Hawaiian and Other Pacific Islander alone",
    "Some Other Race alone",
    "White; Black or African American",
    "White; American Indian and Alaska Native",
    "White; Asian",
    "White; Native Hawaiian and Other Pacific Islander",
    "White; Some Other Race",
    "Black or African American; American Indian and Alaska Native",
    "Black or African American; Asian",
    "Black or African American; Native Hawaiian and Other Pacific Islander",
    "Black or African American; Some Other Race",
    "American Indian and Alaska Native; Asian",
    "American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "American Indian and Alaska Native; Some Other Race",
    "Asian; Native Hawaiian and Other Pacific Islander",
    "Asian; Some Other Race",
    "Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native",
    "White; Black or African American; Asian",
    "White; Black or African American; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; Some Other Race",
    "White; American Indian and Alaska Native; Asian",
    "White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "White; American Indian and Alaska Native; Some Other Race",
    "White; Asian; Native Hawaiian and Other Pacific Islander",
    "White; Asian; Some Other Race",
    "White; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Asian",
    "Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "Black or African American; American Indian and Alaska Native; Some Other Race",
    "Black or African American; Asian; Native Hawaiian and Other Pacific Islander",
    "Black or African American; Asian; Some Other Race",
    "Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "American Indian and Alaska Native; Asian; Some Other Race",
    "American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Asian",
    "White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; American Indian and Alaska Native; Some Other Race",
    "White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; Asian; Some Other Race",
    "White; Black or African American; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "White; American Indian and Alaska Native; Asian; Some Other Race",
    "White; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "Black or African American; American Indian and Alaska Native; Asian; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander",
    "White; Black or African American; American Indian and Alaska Native; Asian; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
    "White; Black or African American; American Indian and Alaska Native; Asian; Native Hawaiian and Other Pacific Islander; Some Other Race",
]

id_to_race = [CensusRace.parse_census_race(race_str) for race_str in _race_strs]
race_to_id = {race: id for (id, race) in enumerate(id_to_race)}
