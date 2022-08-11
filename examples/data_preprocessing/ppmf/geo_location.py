from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True, eq=True)
class GeoLocation:
    state_id: int = None
    county_id: int = None
    census_tract: int = None
    block: int = None

    def contains(self, other: GeoLocation) -> bool:
        if self.state_id != other.state_id and self.state_id is not None:
            return False
        if self.county_id != other.county_id and self.county_id is not None:
            return False
        if self.census_tract != other.census_tract and self.census_tract is not None:
            return False
        if self.block != other.block and self.block is not None:
            return False
        return True

    def get_state(self):
        return GeoLocation(self.state_id)

    def get_county(self):
        return GeoLocation(self.state_id, self.county_id)

    def get_census_tract(self):
        return GeoLocation(self.state_id, self.county_id, self.census_tract)

    def set_state_id(self, state_id: int) -> GeoLocation:
        return GeoLocation(state_id, self.county_id, self.census_tract, self.block)

    def set_county_id(self, county_id: int) -> GeoLocation:
        return GeoLocation(self.state_id, county_id, self.census_tract, self.block)

    def set_census_tract(self, census_tract: int) -> GeoLocation:
        return GeoLocation(self.state_id, self.county_id, census_tract, self.block)

    def set_block(self, block: int) -> GeoLocation:
        return GeoLocation(self.state_id, self.county_id, self.census_tract, block)

    def to_gisjoin(self) -> str:
        result = "G"
        if self.state_id is not None:
            result += f"{self.state_id:02}0"
        if self.county_id is not None:
            result += f"{self.county_id:03}0"
        if self.census_tract is not None:
            result += f"{self.census_tract:06}"
        if self.block is not None:
            result += f"{self.block:04}"
        return result

    def to_geoid(self) -> str:
        result = ""
        if self.state_id is not None:
            result += f"{self.state_id:02}"
        if self.county_id is not None:
            result += f"{self.county_id:03}"
        if self.census_tract is not None:
            result += f"{self.census_tract:06}"
        if self.block is not None:
            result += f"{self.block:04}"
        return result

    @staticmethod
    def parse_gisjoin(gisjoin_str: str) -> GeoLocation:
        # G 420 0010 030101 1000
        # 0 123 4567 890123 4567
        state_id = None
        county_id = None
        census_tract = None
        block = None
        if len(gisjoin_str) >= 4:
            state_id = int(gisjoin_str[1:3])
        if len(gisjoin_str) >= 8:
            county_id = int(gisjoin_str[4:7])
        if len(gisjoin_str) >= 14:
            census_tract = int(gisjoin_str[8:14])
        if len(gisjoin_str) >= 18:
            block = int(gisjoin_str[14:18])
        return GeoLocation(state_id, county_id, census_tract, block)

    @staticmethod
    def parse_geoid(geoid_str: str) -> GeoLocation:
        state_id = None
        county_id = None
        census_tract = None
        block = None
        if len(geoid_str) >= 2:
            state_id = int(geoid_str[0:2])
        if len(geoid_str) >= 5:
            county_id = int(geoid_str[2:5])
        if len(geoid_str) >= 11:
            census_tract = int(geoid_str[5:11])
        if len(geoid_str) >= 15:
            block = int(geoid_str[11:15])
        return GeoLocation(state_id, county_id, census_tract, block)

