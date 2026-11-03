import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest

from pr_agent.algo import output_sinks, run_output
from pr_agent.config_loader import get_settings
from pr_agent.git_providers.azuredevops_provider import AzureDevopsProvider
from pr_agent.git_providers.bitbucket_provider import BitbucketProvider
from pr_agent.git_providers.bitbucket_server_provider import BitbucketServerProvider
from pr_agent.git_providers.codecommit_provider import CodeCommitProvider
from pr_agent.git_providers.gerrit_provider import GerritProvider
from pr_agent.git_providers.gitea_provider import GiteaProvider
from pr_agent.git_providers.github_provider import GithubProvider
from pr_agent.git_providers.gitlab_provider import GitLabProvider
from pr_agent.git_providers.local_git_provider import LocalGitProvider
from pr_agent.git_providers.plain_diff_provider import PlainDiffGitProvider
from pr_agent.tools import pr_code_suggestions, pr_description
from pr_agent.tools.pr_reviewer import PRReviewer
from tests.unittest._settings_helpers import restore_settings, snapshot_settings

PR = {
    "url": "https://github.com/org/repo/pull/42",
    "provider": "github",
    "repo": "org/repo",
    "number": 42,
    "head_sha": "abc123",
}
EMPTY_PR = dict.fromkeys(PR)


@pytest.fixture
def configured_sinks(monkeypatch, tmp_path):
    snapshot = snapshot_settings(("push_outputs",))
    path = tmp_path / "outputs.jsonl"
    get_settings().set("push_outputs", {
        "enable": True,
        "channels": ["stdout", "file", "webhook", "slack", "telegram"],
        "file_path": str(path),
        "webhook_url": "https://example.test/webhook",
        "slack_webhook_url": "https://example.test/slack",
        "telegram_bot_token": "test:token",
        "telegram_chat_id": "chat",
    })
    posts = []
    monkeypatch.setattr(output_sinks.requests, "post", lambda url, **kw: (
        posts.append((url, kw["json"])) or SimpleNamespace(status_code=200)
    ))
    try:
        yield path, posts
    finally:
        restore_settings(snapshot)


@pytest.mark.parametrize("message_type", ["review", "describe", "improve"])
async def test_all_sinks_receive_the_same_pr_identity(configured_sinks, capsys, message_type):
    path, posts = configured_sinks
    provider = SimpleNamespace(get_pr_identity=lambda: PR.copy())

    await run_output.push_outputs_async(message_type, payload={"result": 1}, markdown="Result", git_provider=provider)

    record = json.loads(capsys.readouterr().out)
    assert record["pr"] == PR
    assert record["type"] == message_type
    assert record["payload"] == {"result": 1}
    assert record["markdown"] == "Result"
    assert json.loads(path.read_text()) == record
    assert posts[0][1] == record
    assert posts[1][1] == {"text": f"Result\n\n{PR['url']}"}
    assert posts[2][1] == {"chat_id": "chat", "text": f"Result\n\n{PR['url']}"}


def test_calls_without_a_provider_keep_working(configured_sinks, capsys):
    _path, posts = configured_sinks

    run_output.push_outputs("review", markdown="Result")

    assert json.loads(capsys.readouterr().out)["pr"] == EMPTY_PR
    assert posts[1][1] == {"text": "Result"}
    assert posts[2][1] == {"chat_id": "chat", "text": "Result"}


def test_a_failed_identity_lookup_does_not_drop_the_output(configured_sinks, capsys, monkeypatch):
    _path, posts = configured_sinks
    provider = SimpleNamespace(get_pr_identity=Mock(side_effect=RuntimeError("private provider details")))
    warnings = []
    monkeypatch.setattr(run_output, "get_logger", lambda: SimpleNamespace(warning=warnings.append))

    run_output.push_outputs("review", markdown="Result", git_provider=provider)

    assert json.loads(capsys.readouterr().out)["pr"] == EMPTY_PR
    assert len(posts) == 3
    assert all("private provider details" not in warning for warning in warnings)


def test_disabled_sinks_do_not_read_provider_metadata(configured_sinks, capsys):
    path, posts = configured_sinks
    get_settings().set("push_outputs.enable", False)
    provider = SimpleNamespace(get_pr_identity=Mock(side_effect=AssertionError("metadata must not be read")))

    run_output.push_outputs("review", markdown="Result", git_provider=provider)

    provider.get_pr_identity.assert_not_called()
    assert not path.exists()
    assert posts == []
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("markdown", [None, "", "Result"])
def test_chat_sinks_append_the_url_to_markdown_or_payload(configured_sinks, capsys, markdown):
    _path, posts = configured_sinks
    provider = SimpleNamespace(get_pr_identity=lambda: PR.copy())

    run_output.push_outputs("review", payload={"result": 1}, markdown=markdown, git_provider=provider)

    text = json.dumps({"result": 1}) if markdown is None else markdown
    assert posts[1][1]["text"] == f"{text}\n\n{PR['url']}"
    assert posts[2][1]["text"] == f"{text}\n\n{PR['url']}"


@pytest.mark.parametrize("body", ["x" * 5000, "😀" * 3000, "x" * 4051 + "😀" * 10])
def test_telegram_keeps_the_pr_link_when_truncating_a_long_message(configured_sinks, body):
    _path, posts = configured_sinks
    provider = SimpleNamespace(get_pr_identity=lambda: PR.copy())

    run_output.push_outputs("review", markdown=body, git_provider=provider)

    text = posts[2][1]["text"]
    assert text.endswith(f"\n\n{PR['url']}")
    assert len(text.encode("utf-16-le")) <= 8192
    assert posts[1][1]["text"] == f"{body}\n\n{PR['url']}"


def test_an_old_custom_provider_can_still_emit(configured_sinks, capsys):
    run_output.push_outputs("review", markdown="Result", git_provider=SimpleNamespace())

    assert json.loads(capsys.readouterr().out)["pr"] == EMPTY_PR


@pytest.mark.parametrize("url", ["/private/local/path", "not-a-url", "javascript:alert(1)", "https://[broken"])
def test_invalid_pr_links_do_not_discard_other_identity_fields(configured_sinks, capsys, url):
    provider = SimpleNamespace(get_pr_identity=lambda: {**PR, "url": url})

    run_output.push_outputs("review", markdown="Result", git_provider=provider)

    assert json.loads(capsys.readouterr().out)["pr"] == {**PR, "url": None}


@pytest.mark.parametrize("number, expected", [("42", 42), (None, None), (False, None), (0, None), (-1, None)])
def test_unknown_or_string_numbers_are_serialized_consistently(configured_sinks, capsys, number, expected):
    provider = SimpleNamespace(get_pr_identity=lambda: {**PR, "number": number})

    run_output.push_outputs("review", git_provider=provider)

    assert json.loads(capsys.readouterr().out)["pr"] == {**PR, "number": expected}


@pytest.mark.parametrize("provider_class, provider_id, fields, url", [
    (GithubProvider, "github", {
        "repo": "org/repo", "pr_num": 42,
        "pr": SimpleNamespace(html_url=PR["url"], head=SimpleNamespace(sha="abc123")),
    }, PR["url"]),
    (GitLabProvider, "gitlab", {
        "gitlab_url": "https://example.test/gitlab", "id_project": "123", "id_mr": 42,
        "mr": SimpleNamespace(web_url="https://example.test/gitlab/org/repo/-/merge_requests/42"),
        "diff_files": [], "_diff_head_sha": "abc123",
    }, "https://example.test/gitlab/org/repo/-/merge_requests/42"),
    (AzureDevopsProvider, "azure", {
        "workspace_slug": "org", "repo_slug": "repo", "pr_num": 42,
        "pr": SimpleNamespace(last_merge_commit=SimpleNamespace(commit_id="abc123")),
    }, "https://example.test/project/_git/repo/pullrequest/42"),
    (BitbucketProvider, "bitbucket", {
        "workspace_slug": "org", "repo_slug": "repo", "pr_num": 42,
        "pr": SimpleNamespace(data={"source": {"commit": {"hash": "abc123"}}}),
    }, "https://bitbucket.org/org/repo/pull-requests/42"),
    (BitbucketServerProvider, "bitbucket_server", {
        "workspace_slug": "org", "repo_slug": "repo", "pr_num": 42,
        "pr": SimpleNamespace(fromRef={"latestCommit": "abc123"}),
    }, "https://example.test/projects/org/repos/repo/pull-requests/42"),
    (GiteaProvider, "gitea", {
        "owner": "org", "repo": "repo", "pr_number": 42, "_base_url_html": "https://example.test",
        "last_commit": SimpleNamespace(sha="abc123"),
    }, "https://example.test/org/repo/pulls/42"),
    (CodeCommitProvider, "codecommit", {
        "repo_name": "repo", "pr_num": 42, "pr": SimpleNamespace(source_commit="abc123"),
    }, "https://example.test/codesuite/codecommit/repositories/repo/pull-requests/42"),
    (GerritProvider, "gerrit", {
        "project": "org/repo", "refspec": "refs/changes/42/42/1",
        "repo": SimpleNamespace(head=SimpleNamespace(commit=SimpleNamespace(hexsha="abc123"))),
    }, "https://example.test/gerrit/c/org/repo/+/42"),
])
def test_real_providers_use_their_cached_pr_snapshot(provider_class, provider_id, fields, url):
    provider = provider_class.__new__(provider_class)
    provider.__dict__.update(fields)
    provider.pr_url = "https://example.test/gerrit" if provider_id == "gerrit" else url

    assert provider.get_pr_identity() == {
        "url": url,
        "provider": provider_id,
        "repo": "repo" if provider_id == "codecommit" else "org/repo",
        "number": 42,
        "head_sha": "abc123",
    }


@pytest.mark.parametrize("provider_class, provider_id", [
    (LocalGitProvider, "local"), (PlainDiffGitProvider, "plain-diff"),
])
def test_unhosted_providers_report_no_hosted_pr(provider_class, provider_id):
    provider = provider_class.__new__(provider_class)

    assert provider.get_pr_identity() == {**EMPTY_PR, "provider": provider_id}


def test_one_unavailable_provider_field_does_not_hide_the_rest():
    provider = GithubProvider.__new__(GithubProvider)
    provider.repo = "org/repo"
    provider.pr_num = 42
    provider.get_pr_url = Mock(side_effect=RuntimeError("private provider details"))
    provider.get_pr_head_sha = lambda: "abc123"

    assert provider.get_pr_identity() == {**PR, "url": None}


@pytest.fixture
def tool_settings():
    overrides = {
        "config.publish_output": True,
        "config.publish_output_progress": False,
        "config.is_auto_command": False,
        "config.propagate_tool_errors": True,
        "pr_description.enable_semantic_files_types": False,
        "pr_description.publish_labels": False,
        "pr_description.use_description_markers": False,
        "pr_description.enable_help_comment": False,
        "pr_description.enable_help_text": False,
    }
    snapshot = snapshot_settings(tuple(overrides))
    for key, value in overrides.items():
        get_settings().set(key, value)
    try:
        yield
    finally:
        restore_settings(snapshot)


@pytest.mark.parametrize("command", ["review", "describe", "improve", "improve-empty"])
async def test_tool_publish_paths_deliver_pr_identity_to_real_sinks(
    configured_sinks, tool_settings, monkeypatch, capsys, command,
):
    provider = MagicMock()
    provider.get_pr_identity.return_value = PR.copy()
    provider.is_supported.return_value = False
    provider.get_files.return_value = ["app.py"]
    if command == "review":
        tool = PRReviewer.__new__(PRReviewer)
        tool.git_provider = provider
        tool._pending_sink_output = ("review", {"summary": "Result"}, "Review")
        await tool._deliver_sink_output()
    elif command == "describe":
        tool = pr_description.PRDescription.__new__(pr_description.PRDescription)
        tool.git_provider = provider
        tool.pr_id = "org/repo/42"
        tool.vars = {}
        tool.prediction = "generated"
        tool.data = None
        tool.file_label_dict = None
        tool._prepare_data = Mock()
        tool._prepare_pr_answer = Mock(return_value=("Title", "Description", ""))
        monkeypatch.setattr(pr_description, "extract_and_cache_pr_tickets", AsyncMock())
        monkeypatch.setattr(pr_description, "retry_with_fallback_models", AsyncMock())
        await tool.run()
    else:
        tool = pr_code_suggestions.PRCodeSuggestions.__new__(pr_code_suggestions.PRCodeSuggestions)
        tool.git_provider = provider
        tool.pr_url = PR["url"]
        tool.progress_response = None
        tool._output_published = False
        tool.is_extended = False
        suggestions = [] if command == "improve-empty" else [{
            "relevant_file": "app.py", "relevant_lines_start": 1, "relevant_lines_end": 1,
            "label": "possible issue", "score": 8, "one_sentence_summary": "Check the input",
            "suggestion_content": "Validate the input before using it.",
        }]
        monkeypatch.setattr(pr_code_suggestions, "retry_with_fallback_models", AsyncMock(return_value={
            "code_suggestions": suggestions,
        }))
        await tool.run()

    record = json.loads(capsys.readouterr().out)
    assert record["pr"] == PR
    assert record["type"] == command.removesuffix("-empty")
    path, posts = configured_sinks
    assert json.loads(path.read_text()) == record
    assert posts[0][1]["pr"] == PR
    assert posts[1][1]["text"].endswith(PR["url"])
    assert posts[2][1]["text"].endswith(PR["url"])
