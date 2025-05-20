import abc
import json
import os
from typing import Dict, List

import networkx as nx
import pandas as pd

from am_combiner.features.article import Features
from am_combiner.features.common import ArticleVisitor


class GraphBasedGeoResolverVisitor:

    """
    GeoResolver visitor interface base class.

    Inheriting classes are supposed to be adding more structural information about
    geographical entities.
    """

    @staticmethod
    def _ensure_path(path):
        if not os.path.exists(path):
            raise ValueError(f"{path} does not exist!")

    @abc.abstractmethod
    def visit_geo_resolver(self, geo_resolver):
        """
        Abstract interface for geo resolver visiting.

        Parameters
        ----------
        geo_resolver: GraphBasedGeoResolver
            A geo resolver object holding information about geographical hierarchies.

        Returns
        -------
        None

        """
        return


class CountriesListGraphBasedGeoResolverVisitor(GraphBasedGeoResolverVisitor):

    """

    Concrete implementation of the ABC.

    This particular object adds information about countries primary names.

    """

    def __init__(self, country_list_fn):
        CountriesListGraphBasedGeoResolverVisitor._ensure_path(country_list_fn)
        all_countries = pd.read_csv(country_list_fn, keep_default_na=False)
        all_countries["Name"] = all_countries["Name"].str.lower()
        all_countries["Code"] = all_countries["Code"].str.lower()
        all_countries.set_index("Name", inplace=True)
        self.all_countries = all_countries

    def visit_geo_resolver(self, geo_resolver):
        """
        Concrete implementation of ABC abstract method.

        Parameters
        ----------
        geo_resolver: GraphBasedGeoResolver
            A geo resolver object holding information about geographical hierarchies.

        Returns
        -------
        None

        """
        for country_name in self.all_countries.index:
            if geo_resolver.resolve_geo_name(country_name, properties={"final": True}):
                continue
            geo_resolver.G.add_node(country_name, final=True)


class CountriesCodesVisitor(CountriesListGraphBasedGeoResolverVisitor):

    """

    Concrete implementation of the ABC.

    This particular object adds information about countries international codes.

    """

    def visit_geo_resolver(self, geo_resolver):
        """
        Concrete implementation of ABC abstract method.

        Parameters
        ----------
        geo_resolver: GraphBasedGeoResolver
            A geo resolver object holding information about geographical hierarchies.

        Returns
        -------
        None

        """
        for country_name in self.all_countries.index:
            resolution = geo_resolver.resolve_geo_name(country_name, properties={"final": True})
            if not resolution:
                print(f"Codes visitor could not resolve the country {country_name}")
                continue
            code = self.all_countries.loc[country_name]["Code"]
            geo_resolver.G.add_node(code, code=True)
            geo_resolver.G.add_edge(code, resolution[0])
            geo_resolver.G.add_edge(resolution[0], code)


class CountriesAliasesGraphBasedGeoResolverVisitor(GraphBasedGeoResolverVisitor):

    """

    Concrete implementation of the ABC.

    This particular object adds information about countries alternative names/spellings.

    """

    NAME = "Name"
    ALIASES = "Aliases"

    def __init__(self, country_aliases_fn):
        CountriesAliasesGraphBasedGeoResolverVisitor._ensure_path(country_aliases_fn)
        countries_aliases = pd.read_csv(country_aliases_fn, delimiter="\t")
        countries_aliases[self.NAME] = countries_aliases[self.NAME].str.lower()
        countries_aliases[self.ALIASES] = countries_aliases[self.ALIASES].apply(
            lambda l: set(l.lower().split(","))
        )
        countries_aliases.set_index(self.NAME, inplace=True)
        self.countries_aliases = countries_aliases

    def visit_geo_resolver(self, geo_resolver):
        """
        Concrete implementation of ABC abstract method.

        Parameters
        ----------
        geo_resolver: GraphBasedGeoResolver
            A geo resolver object holding information about geographical hierarchies.

        Returns
        -------
        None

        """
        for ct, country in enumerate(self.countries_aliases.index):
            # for ct, country in enumerate(customs):
            resolved = False
            if not geo_resolver.resolve_geo_name(country, properties={"final": True}):
                # If we could not resolve the original country name, we try to resolve
                # all aliases. If we can resolve an alias, we put the original `original`
                # name into the list of aliases and the resolved alias becomes a new primary
                # country name, then add this new node as a new graph
                for alias in self.countries_aliases.loc[country][self.ALIASES]:
                    if geo_resolver.resolve_geo_name(alias, properties={"final": True}):
                        self.countries_aliases.loc[country][self.ALIASES].add(country)
                        self.countries_aliases.loc[country][self.ALIASES].remove(alias)
                        self.countries_aliases.rename(index={country: alias}, inplace=True)
                        geo_resolver.G.add_node(alias, final=True)
                        country = alias
                        resolved = True
                        break
                if not resolved:
                    print(f"{self.__class__.__name__} could not resolve country name {country}")

            for alias in self.countries_aliases.loc[country][self.ALIASES]:
                alias_name = alias
                geo_resolver.G.add_edge(alias_name, country)


class CountyAdditionGraphBasedGeoResolverVisitor(GraphBasedGeoResolverVisitor):

    """

    Concrete implementation of the ABC.

    This particular object adds information about counties/region levels capitals.

    """

    def __init__(self, resource_path):
        CountyAdditionGraphBasedGeoResolverVisitor._ensure_path(resource_path)
        self.generic_data_folder = os.path.join(resource_path, "data")
        CountyAdditionGraphBasedGeoResolverVisitor._ensure_path(self.generic_data_folder)
        self.divisions_data_folder = os.path.join(resource_path, "divisions")
        CountyAdditionGraphBasedGeoResolverVisitor._ensure_path(self.divisions_data_folder)

    def visit_geo_resolver(self, geo_resolver):
        """
        Concrete implementation of ABC abstract method.

        Parameters
        ----------
        geo_resolver: GraphBasedGeoResolver
            A geo resolver object holding information about geographical hierarchies.

        Returns
        -------
        None

        """
        for file in os.listdir(self.divisions_data_folder):
            code, _ = os.path.splitext(file)
            resolution = geo_resolver.resolve_geo_name(code, properties={"final": True})
            if not resolution:
                print(f"{self.__class__.__name__} could not find country for code {code}")
            # print(f'County visitor could resolved the country {code} to {resolution[0]}')
            full_fn = os.path.join(self.divisions_data_folder, file)
            with open(full_fn, "r") as f:
                data = json.load(f)
            for v in data:
                name_ = v["name"]
                if name_ is None:
                    continue
                lower = name_.lower()
                geo_resolver.G.add_node(lower, type="state")
                geo_resolver.G.add_edge(resolution[0], lower)
                geo_resolver.G.add_edge(lower, resolution[0])


class CapitalAdditionVisitor(CountyAdditionGraphBasedGeoResolverVisitor):

    """

    Concrete implementation of the ABC.

    This particular object adds information about countries capitals.

    """

    def visit_geo_resolver(self, geo_resolver):
        """
        Concrete implementation of ABC abstract method.

        Parameters
        ----------
        geo_resolver: GraphBasedGeoResolver
            A geo resolver object holding information about geographical hierarchies.

        Returns
        -------
        None

        """
        for file in os.listdir(self.generic_data_folder):
            code, _ = os.path.splitext(file)
            resolution = geo_resolver.resolve_geo_name(code, properties={"final": True})
            if not resolution:
                print(f"{self.__class__.__name__} could not find country for code {code}")
                continue
            # print(f'County visitor could resolved the country {code} to {resolution[0]}')
            full_fn = os.path.join(self.generic_data_folder, file)
            with open(full_fn, "r") as f:
                data = json.load(f)
            capital_ = data["capital"]
            if capital_ is None:
                continue
            lower = capital_.lower()
            geo_resolver.G.add_node(lower, type="capital")
            geo_resolver.G.add_edge(resolution[0], lower)
            geo_resolver.G.add_edge(lower, resolution[0])


class GraphBasedGeoResolver:

    """

    Resolves given geographical entities to other geographical entities somehow related to them.

    Attributes
    ----------
        G: nx.DiGraph

    """

    def __init__(self):
        self.G = nx.DiGraph()

    def accept_visitor(self, visitor):
        """
        Accept a visitor for the class.

        Parameters
        ----------
        visitor: GraphBasedGeoResolverVisitor
            A visitor instance

        Returns
        -------
        None

        """
        visitor.visit_geo_resolver(self)

    def resolve_geo_name(self, geo_name: str, properties: Dict = None) -> List[str]:
        """
        Attempt to resolve a given geo name to some other geographical instance related to it.

        Parameters
        ----------
        geo_name: str
            Name to be resolved
        properties: Dict
            Resolution parameters. TODO Add format description

        Returns
        -------
        List[str]
            A list of geographical names the given string was resolved to.

        """
        if not properties:
            raise ValueError("Properties must not be empty")
        geo_name = geo_name.lower()
        if not self.G.has_node(geo_name):
            return []

        resolutions = []
        for node in nx.dfs_preorder_nodes(self.G, source=geo_name):
            match = True
            for k, v in properties.items():
                if k not in self.G.nodes[node]:
                    match = False
                    break
                if self.G.nodes[node][k] != v:
                    match = False
                    break
            if not match:
                continue
            resolutions.append(node)
            if resolutions:
                break
        return resolutions


class ArticleGeoVisitor(ArticleVisitor):

    """

    Concrete implementation of the ABC. Enriches Article objects with geographical information.

    Attributes
    ----------
    geo_resolver: GraphBasedGeoResolver
        A GraphBasedGeoResolver object populated with geographical hierarchical information.
    target_feature: str
        Name of the fields to take extracted information from.

    """

    def __init__(self, geo_resolver: GraphBasedGeoResolver = None):
        super().__init__()
        if not geo_resolver:
            geo_resolver = get_full_geo_resolver()
        self.geo_resolver = geo_resolver
        self.source_feature = Features.GPE
        self.target_feature = Features.GPE_CLEAN

    def visit_article(self, article):
        """
        Concrete implementation of the ABC method.

        Parameters
        ----------
        article: Article
            An article object to resolve geographical information for

        Returns
        -------
        None

        """
        if self.source_feature not in article.extracted_entities:
            return
        new_state = set()
        resolved_this_article = 0
        for entity in article.extracted_entities[self.source_feature]:
            entity_str = str(entity).strip()
            resolution = self.geo_resolver.resolve_geo_name(entity_str, {"final": True})
            if not resolution:
                new_state.add(entity_str)
            else:
                resolved_this_article += 1
                new_state.add(resolution[0])
        article.extracted_entities[self.target_feature] = new_state


def get_full_geo_resolver():
    """Build georesolver with all available features."""
    geo_resolver = GraphBasedGeoResolver()
    country_list_visitor = CountriesListGraphBasedGeoResolverVisitor(
        country_list_fn="am_combiner/data/geo/all_countries.csv"
    )
    geo_resolver.accept_visitor(country_list_visitor)

    country_aliases_visitor = CountriesAliasesGraphBasedGeoResolverVisitor(
        country_aliases_fn="am_combiner/data/geo/countries_alternative_names.tsv"
    )
    geo_resolver.accept_visitor(country_aliases_visitor)

    countries_codes_visitor = CountriesCodesVisitor(
        country_list_fn="am_combiner/data/geo/all_countries.csv"
    )
    geo_resolver.accept_visitor(countries_codes_visitor)

    county_addition_visitor = CountyAdditionGraphBasedGeoResolverVisitor(
        resource_path="am_combiner/data/geo/"
    )
    geo_resolver.accept_visitor(county_addition_visitor)

    capital_addition_visitor = CapitalAdditionVisitor(resource_path="am_combiner/data/geo/")
    geo_resolver.accept_visitor(capital_addition_visitor)
    return geo_resolver
