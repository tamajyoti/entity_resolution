import datetime

import regex as re

from am_combiner.features.article import Article, Features
from am_combiner.features.common import ArticleVisitor


class TimeStandardisationVisitor(ArticleVisitor):

    """
    A concrete implementation of the ArticleVisitor class.

    Standardises the time in string format.

    """

    def __init__(self) -> None:
        super().__init__()
        self.time_field = Features.TIME
        self.target_field = Features.TIME_CLEAN
        self.patterns = ["%I %p", "%I%p", "%I:%M %p", "%I:%M%p", "%H:%M"]
        self.re_patterns = [re.compile(r"\d{1,2}:?(?:\d{2})?\s?(?:AM|PM|am|pm)")]

    @staticmethod
    def _normalise_am_pm(ampm: str) -> str:
        """
        Take a string that supposedly contains some representation of time and normalise it.

        It normalises it so that it could later be parsed by standard python libraries.
        Specifically, it substitutes am/pm/a.m./p.m. with AM/PM.

        Parameters
        ----------
        ampm:
            A string that supposedly represents time. Does not have to be in AM/PM format.

        Returns
        -------
            If there were any entries of am/pm, they will be normalised so that python datetime
            library could parse them.

        """
        ret = ampm.replace("p.m.", "PM")
        ret = ret.replace("a.m.", "AM")
        ret = ret.replace("pm", "PM")
        ret = ret.replace("am", "AM")

        return ret

    def _attempt_parsing(self, time_string: str) -> str:
        """
        Attempt to create python datetime object from a string.

        For that a very limited number of patterns is used.

        Parameters
        ----------
        time_string:
            A string that supposedly contains some time representation.

        Returns
        -------
            Time extracted from a parsed date.

        """
        result = None
        for p in self.patterns:
            try:
                result = datetime.datetime.strptime(time_string, p)
                # If date was parsed successfully,
                # time can be extracted since that was the only information available
                result = str(result.time())
            except ValueError:
                pass
        return result

    def visit_article(self, article: Article) -> None:
        """
        Visit the article and apply the time standardisation.

        Parameters
        ----------
        article:
            The article to be visited.

        """
        # TODO this whole logic should live in a separate class that can be used by anyone else.
        #  Visitors should just invoke it, nothing else
        if self.time_field not in article.extracted_entities:
            return
        extracted_time = [t.text for t in article.extracted_entities[self.time_field]]
        # with this we will replace the time field in article
        visitor_results = set()
        for time in extracted_time:
            new_time = self._normalise_am_pm(time)
            result = self._attempt_parsing(new_time)
            if not result:
                # If direct parsing did not help, try to find some time strings with regex patterns
                # TODO 3.17AM is recognized as 17AM
                # TODO regex parsing for in-string "time was 23:32"
                for p in self.re_patterns:
                    result = re.findall(p, new_time)
                    if result:
                        for ct, t in enumerate(result):
                            parsed = self._attempt_parsing(t)
                            # if we manage to extract stuff,
                            # replace old representation with the new parsed one
                            if parsed:
                                result[ct] = parsed
                        # no need to keep trying if we've just parsed something
                        break
            if result:
                if isinstance(result, list):
                    visitor_results.update(result)
                else:
                    visitor_results.add(result)
        article.extracted_entities[self.target_field] = visitor_results
