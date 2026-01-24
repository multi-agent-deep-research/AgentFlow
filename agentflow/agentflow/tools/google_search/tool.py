import os
import json
import re
from typing import Dict, List, Optional
from html.parser import HTMLParser

# pip install yandex-cloud-ml-sdk
from yandex_cloud_ml_sdk import YCloudML
from dotenv import load_dotenv

from agentflow.tools.base import BaseTool

load_dotenv()

TOOL_NAME = "Ground_Google_Search_Tool"

LIMITATIONS = """
1. This tool is only suitable for general information search.
2. Results are limited to what Yandex Search returns via its API.
3. It is not suitable for searching and analyzing videos on YouTube or other video platforms.
"""

BEST_PRACTICES = """
1. Choose this tool when you want general information about a topic.
2. Use concise queries; avoid long, multi-part questions.
3. Validate critical facts by opening the source URLs.
"""

USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36"


class SearchResultParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.in_result = False
        self.in_title = False
        self.in_snippet = False
        self.current_result = {}

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "li" and attrs_dict.get("class") == "serp-item":
            self.in_result = True
            self.current_result = {}
        elif self.in_result:
            if tag == "a" and "href" in attrs_dict:
                self.current_result["url"] = attrs_dict["href"]
            if tag == "h2":
                self.in_title = True
                self.current_result["title"] = ""
            if tag == "div" and "snippet" in attrs_dict.get("class", ""):
                self.in_snippet = True
                self.current_result["snippet"] = ""

    def handle_endtag(self, tag):
        if tag == "li" and self.in_result:
            if self.current_result:
                self.results.append(self.current_result)
            self.in_result = False
            self.current_result = {}
        elif tag == "h2" and self.in_title:
            self.in_title = False
        elif tag == "div" and self.in_snippet:
            self.in_snippet = False

    def handle_data(self, data):
        if self.in_title:
            self.current_result["title"] = self.current_result.get("title", "") + data.strip()
        elif self.in_snippet:
            self.current_result["snippet"] = self.current_result.get("snippet", "") + data.strip()


class Google_Search_Tool(BaseTool):
    def __init__(self):
        super().__init__(
            tool_name=TOOL_NAME,
            tool_description="A web search tool powered by Yandex Search API that provides real-time information from the internet.",
            tool_version="1.0.0",
            input_types={
                "query": "str - The search query to find information on the web.",
                "add_citations": "bool - Whether to add citations to the results. If True, the results will be formatted with citations. By default, it is True.",
            },
            output_type="str - The search results of the query.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(query="What is the capital of France?")',
                    "description": "Search for general information about the capital of France with default citations enabled.",
                },
                {
                    "command": 'execution = tool.execute(query="Who won the euro 2024?", add_citations=False)',
                    "description": "Search for information about Euro 2024 winner without citations.",
                },
                {
                    "command": 'execution = tool.execute(query="Physics and Society article arXiv August 11, 2016", add_citations=True)',
                    "description": "Search for specific academic articles with citations enabled.",
                },
            ],
            user_metadata={
                "limitations": LIMITATIONS,
                "best_practices": BEST_PRACTICES,
            },
        )
        self.folder_id = os.getenv("YANDEX_SEARCH_FOLDER_ID")
        self.api_key = os.getenv("YANDEX_SEARCH_API_KEY")
        self.search_type = os.getenv("YANDEX_SEARCH_TYPE", "en")
        self.poll_interval = int(os.getenv("YANDEX_SEARCH_POLL_INTERVAL", "1"))
        print(self.api_key)

        if not self.folder_id or not self.api_key:
            raise Exception(
                "Yandex Search credentials not found. Please set YANDEX_SEARCH_FOLDER_ID and "
                "YANDEX_SEARCH_API_KEY environment variables."
            )

        self.sdk = YCloudML(
            folder_id=self.folder_id,
            auth=self.api_key,
        )

    def _parse_html_results(self, html_content: str) -> List[Dict[str, str]]:
        url_pattern = r'<a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
        snippet_pattern = r'<div\s+[^>]*class="[^"]*snippet[^"]*"[^>]*>(.*?)</div>'

        results = []
        matches = re.finditer(url_pattern, html_content, re.DOTALL | re.IGNORECASE)

        for match in matches:
            url = match.group(1)
            title_html = match.group(2)
            title = re.sub("<.*?>", "", title_html).strip()

            if url.startswith("http") and title:
                result = {"title": title, "url": url, "snippet": ""}

                snippet_match = re.search(
                    snippet_pattern, html_content[match.end() : match.end() + 500], re.DOTALL | re.IGNORECASE
                )
                if snippet_match:
                    snippet_html = snippet_match.group(1)
                    result["snippet"] = re.sub("<.*?>", "", snippet_html).strip()

                if result not in results:
                    results.append(result)

                if len(results) >= 10:
                    break

        return results

    def _execute_search(self, query: str) -> List[Dict[str, str]]:
        try:
            search = self.sdk.search_api.web(
                search_type=self.search_type,
                user_agent=USER_AGENT,
            )

            operation = search.run_deferred(query, format="html", page=0)
            search_result_bytes = operation.wait(poll_interval=self.poll_interval)
            search_result_html = search_result_bytes.decode("utf-8")

            return self._parse_html_results(search_result_html)

        except Exception as e:
            raise Exception(f"Yandex Search failed: {str(e)}")

    def execute(self, query: str, add_citations: bool = True):
        results = self._execute_search(query)

        if not results:
            return "No results found."

        lines = []
        for idx, result in enumerate(results, start=1):
            title = result.get("title", "Result")
            snippet = result.get("snippet", "").strip()
            url = result.get("url", "")
            if add_citations:
                line = f"{idx}. {title}"
                if snippet:
                    line += f" - {snippet}"
                line += f" [{idx}]({url})"
            else:
                line = f"{idx}. {title}"
                if snippet:
                    line += f" - {snippet}"
                if url:
                    line += f" ({url})"
            lines.append(line)

        return "\n".join(lines)

    def get_metadata(self):
        metadata = super().get_metadata()
        return metadata


if __name__ == "__main__":

    def print_json(result):
        import json

        print(json.dumps(result, indent=4))

    google_search = Google_Search_Tool()

    metadata = google_search.get_metadata()
    print("Tool Metadata:")
    print_json(metadata)

    examples = [
        {"query": "What is the capital of France?", "add_citations": True},
        {"query": "Who won the euro 2024?", "add_citations": False},
        {"query": "Physics and Society article arXiv August 11, 2016", "add_citations": True},
    ]

    for example in examples:
        print(f"\nExecuting search: {example['query']}")
        try:
            result = google_search.execute(**example)
            print("Search Result:")
            print(result)
        except Exception as e:
            print(f"Error: {str(e)}")
        print("-" * 50)

    print("Done!")
