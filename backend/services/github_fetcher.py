from github import Github
from config import GITHUB_TOKEN, TARGET_REPO, MAX_ISSUES, RAW_DIR
import json
 
def fetch_closed_issues(repo_name: str = TARGET_REPO, limit: int = MAX_ISSUES):
    gh = Github(GITHUB_TOKEN, per_page=100)
    repo = gh.get_repo(repo_name)
 
    raw_issues = []
    for issue in repo.get_issues(state='closed', sort='created', direction='desc'):
        if issue.pull_request is not None:
            continue  # skip PRs — see the gotcha above
        if issue.user and issue.user.type == 'Bot':
            continue  # skip bot-authored issues (stale-bot, dependabot, etc.)
 
        raw_issues.append({
            'number': issue.number,
            'title': issue.title,
            'body': (issue.body or '')[:2000],
            'labels': [l.name for l in issue.labels],
            'comments': issue.comments,
        })
        if len(raw_issues) % 50 == 0:
            print(f"  fetched {len(raw_issues)}/{limit}...", flush=True)
        if len(raw_issues) >= limit:
            break
 
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"{repo_name.replace('/', '_')}.jsonl"
    with open(out_path, 'w') as f:
        for item in raw_issues:
            f.write(json.dumps(item) + '\n')

    return str(out_path), len(raw_issues)
