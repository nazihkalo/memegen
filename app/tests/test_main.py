from .. import settings


def describe_index():
    def it_redirects_to_the_docs(expect, client):
        request, response = client.get("/")
        expect(response.status) == 200
        expect(response.text).contains("openapi.json")

    def it_contains_favicon(expect, client):
        request, response = client.get("/favicon.ico")
        expect(response.status) == 200

    def it_contains_robots(expect, client):
        request, response = client.get("/robots.txt")
        expect(response.status) == 200
        expect(response.text).contains("Allow: /\n")


def describe_test():
    def it_redirects_to_the_index(expect, client):
        request, response = client.get("/test", allow_redirects=False)
        expect(response.status) == 302
        expect(response.headers["Location"]) == "/"

    def it_displays_test_images_when_debug(expect, client, monkeypatch):
        monkeypatch.setattr(settings, "DEBUG", True)
        request, response = client.get("/test", allow_redirects=False)
        expect(response.status) == 200
        expect(response.text.count("img")) > 5
        expect(response.text.count("img")) < 100
