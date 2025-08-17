from jira import JIRA


def connect_to_jira(jira_url, email, api_token):
    try:
        jira = JIRA(server=jira_url, basic_auth=(email, api_token))
        return jira
    except Exception as e:
        raise Exception(f"Jira connection failed: {e}")


def get_issue_details(jira, issue_key):
    try:
        issue = jira.issue(issue_key)
        details = {
            "key": issue.key,
            "summary": issue.fields.summary,
            "description": issue.fields.description,
            "assignee": issue.fields.assignee.displayName if issue.fields.assignee else "Unassigned",
            "due_date": issue.fields.duedate,
            "issue_type": issue.fields.issuetype.name,
            "subtasks": [],
            "linked_issues": []
        }

        # Fetch sub-tasks
        if issue.fields.subtasks:
            for subtask in issue.fields.subtasks:
                sub = jira.issue(subtask.key)
                details["subtasks"].append({
                    "key": sub.key,
                    "summary": sub.fields.summary,
                    "description": sub.fields.description,
                    "assignee": sub.fields.assignee.displayName if sub.fields.assignee else "Unassigned",
                    "status": sub.fields.status.name,
                    "due_date": sub.fields.duedate,
                })

        # Fetch linked issues
        if hasattr(issue.fields, "issuelinks"):
            for link in issue.fields.issuelinks:
                linked_issue = None
                if hasattr(link, "outwardIssue"):
                    linked_issue = link.outwardIssue
                elif hasattr(link, "inwardIssue"):
                    linked_issue = link.inwardIssue

                if linked_issue:
                    linked = jira.issue(linked_issue.key)
                    details["linked_issues"].append({
                        "key": linked.key,
                        "summary": linked.fields.summary,
                        "description": linked.fields.description,
                        "assignee": linked.fields.assignee.displayName if linked.fields.assignee else "Unassigned",
                        "status": linked.fields.status.name,
                        "due_date": linked.fields.duedate,
                    })

        return details

    except Exception as e:
        raise Exception(f"Failed to fetch issue details: {e}")
