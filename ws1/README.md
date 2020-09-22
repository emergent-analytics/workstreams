# Regional Risk Notebooks

This folder contains various notebook that compute local risk indices for certain countries and regions.

## Naming geo entries

There are numerous column names that need to be mapped onto a concept of countries and entities below country level.

|Concept|Meaning      |Meanings encountered|schema.org proposal|
|-------|-------------|--------------------|-------------------|
|Country|[Countries or Areas](https://unstats.un.org/unsd/methodology/m49/), the notion of a passport and citizenship applies|There are also Federations and Nations, or Unities|http://schema.orgs/Country|
|State  |An entity below country|States, provinces, regions are common names.|https://schema.org/State|

There is too many standards available for sturcturing and naming these entities. They include 
[ISO 3166](https://www.iso.org/iso-3166-country-codes.html), which is available for free at [IBAN](https://www.iban.com/country-codes),
[United Nations's M49](https://unstats.un.org/unsd/methodology/m49/), or the [European Nomenclature of Territoral Units for Statistics
NUTS](https://ec.europa.eu/eurostat/en/web/nuts/background).

For our assessment, we used the following structure for the countries we consider:

| Country | Name of Level considered | Rank of Level |
|---------|--------------------------|---------------|
| Germany | Landkreise and kreisfreise Städte and Bezirke (Berlin) | Country->States->Regions and Municpipalities |
| Italy   | Reggione | TBD |
| France  | Departements | TBD |
| UK      | UTLA (Upper Tier Local Authorities), NB: England only | Country->Nation->UTLA |
| South Korea | <Provinces> | TBD |

The [NUTS classification](https://ec.europa.eu/eurostat/documents/345175/629341/NUTS2021.xlsx) is a comprehensive summary of the logical
structure of countries in Europe.

