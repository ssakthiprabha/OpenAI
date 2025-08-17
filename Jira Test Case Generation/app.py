from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from jira import JIRA
import openai
import re
from collections import Counter


# Configuration
JIRA_URL = "https://ssakthi.atlassian.net"
JIRA_EMAIL = "s.sakthiprabha@gmail.com"
JIRA_API_TOKEN = "Jira API Token Comes here"
OPENAI_API_KEY = "Open AI key comes here"

client = openai.OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)


# Get Jira Issue Details
def get_issue_details(jira, issue_key):
    issue = jira.issue(issue_key)
    details = {
        "main_description": issue.fields.description,
        "subtasks": [],
        "linked_issues": []
    }

    for sub in issue.fields.subtasks:
        sub_issue = jira.issue(sub.key)
        details["subtasks"].append({
            "key": sub_issue.key,
            "summary": sub_issue.fields.summary,
            "description": sub_issue.fields.description,
            "assignee": sub_issue.fields.assignee.displayName if sub_issue.fields.assignee else "Unassigned",
            "status": sub_issue.fields.status.name,
            "due_date": sub_issue.fields.duedate,
        })

    for link in getattr(issue.fields, "issuelinks", []):
        linked_issue = getattr(link, "outwardIssue", getattr(link, "inwardIssue", None))
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


# Extract Module Mentions for Heatmap
def extract_modules(text):
    modules = [
        "Voyage_T", "Voyage_leg_T", "Container_Segment_T", "Container_T",
        "Delivery_Container_xref", "Delivery_Movement", "Delivery_T",
        "Delivery_line_item_T", "Shipment_T", "Shipment_line_item_T",
        "shipment", "container", "delivery", "customer", "path", "receiving",
        "invoice", "bill", "movement", "paperwork", "quote", "bol","Customer","Carrier"
    ]
    found = re.findall(r'\b(?:' + '|'.join(re.escape(m) for m in modules) + r')\b', text)
    return dict(Counter(found))

# Decrypt the sensitive Data
def redact_sensitive_data(text):
    if not text:
        return ""
    # Replace emails, URLs, names, long IDs, etc.
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.\w+\b', '[REDACTED_EMAIL]', text)
    text = re.sub(r'https?://\S+', '[REDACTED_URL]', text)
    text = re.sub(r'\b[A-Z]{2,10}-\d{1,6}\b', '[REDACTED_ISSUE_KEY]', text)
    return text


# Routes
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    issue_key = data.get("issueKey")

    if not issue_key:
        return jsonify({"error": "No Jira issue key provided."}), 400

    try:
        jira = JIRA(server=JIRA_URL, basic_auth=(JIRA_EMAIL, JIRA_API_TOKEN))
        issue_data = get_issue_details(jira, issue_key)

        prompt = f"Generate test cases for:\n{issue_data['main_description']}\n\n"
        for s in issue_data['subtasks']:
            prompt += f"- {s['summary']} ({s['description']})\n"
        for l in issue_data['linked_issues']:
            prompt += f"- {l['summary']} ({l['description']})\n"

            final_prompt = f"""
You are a senior test automation engineer. Based on the Jira story and related details below, do two things:

1. Generate a test steps for the test case which includes the following information for each test step::
   - The main Jira description
        - Test Step – Description of the test step/action that needs to be performed for this individual step.
        - Test Data – Any data that is required for the test step to be performed.
        - Expected Result – The expected result after performing the actions for this individual step.
   - Any subtasks and linked issues
   - Edge cases and validation points if applicable
   - Any edge cases or validation steps mentioned

2. Generate a **complete Java Selenium TestNG automation script** that automates the test cases described above. The automation code must follow these rules:
    - Create a class named `Test_<issueKey>` (e.g., Test_ABC123).
    - Use `@BeforeClass` to initialize Test WebDriver.
    - Use `@AfterClass` to quit WebDriver.
    - Define **one `@Test` method per test case**.
    - Inside each `@Test`, perform:
        - Element interactions like `click`, `sendKeys`, `select`, etc., using realistic WebDriver locators (`By.id`, `By.name`, `By.xpath`, etc.).
        - Sample input values and sample navigation (e.g., open login page, fill a form, etc.).
        - Assertions (`assertEquals`, `assertTrue`, etc.) to validate outcomes.
    - Include meaningful variable names and real examples of form fields, buttons, etc.
    - Include clear, descriptive comments above every block of code to explain its purpose.
    - Avoid vague lines like “your code here” or “add logic”. Instead, fill in with realistic working examples.
    - Ensure the code compiles and looks like a real automation script written by a senior QA engineer.

Data Flow:
Voyage_T ➝ Voyage_leg_T ➝ Container_Segment_T ➝ Container_T ➝ Delivery_Container_xref ➝ Delivery_Movement ➝ Delivery_T ➝ Delivery_line_item_T ➝ Shipment_T ➝ Shipment_line_item_T

Process Instructions:
1. Quotes (BOL)
2. Incoming
3. Receiving
4. After Receiving
5. Customers: Shipper, Consignee, Walmart, BillTo
6. Paths: Paperwork Path, Delivery Path
7. Container Movement

Details from Jira:
{prompt}

Format your response as follows:

1. A numbered list of test case descriptions.
2. Then, **begin the Selenium automation section with exactly this heading:**

### Selenium Test Automation (Java + TestNG)

3. Under that heading, include the complete and well-commented Java Selenium TestNG code inside triple backticks like this:
```java
// Your code here

"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": final_prompt}]
        )

        response_text = response.choices[0].message.content.strip()

        if "Selenium Test Automation (Java + TestNG)" in response_text:
            parts = response_text.split("Selenium Test Automation (Java + TestNG)")
            test_cases = parts[0].strip()
            automation_snippet = "Selenium Test Automation (Java + TestNG)" + parts[1].strip()
        else:
            test_cases = response_text
            automation_snippet = ""

        # Combine all relevant text for heatmap extraction
        combined_text = (
            (issue_data['main_description'] or '') +
            ''.join(s['description'] or '' for s in issue_data['subtasks']) +
            ''.join(l['description'] or '' for l in issue_data['linked_issues'])
        )

        heatmap_data = extract_modules(combined_text)

        return jsonify({
            "testCases": test_cases,
            "automationSnippet": automation_snippet,
            "jiraDetails": {
                "mainDescription": redact_sensitive_data(issue_data['main_description']),
                "subtasks": [
                            {
                                "key": s.get("key", "undefined"),
                                "summary": s.get("summary", "undefined"),
                                "description": redact_sensitive_data(s.get("description", "No description"))
                            }
                                for s in issue_data.get("subtasks", [])
                            ],
                "linkedIssues": [
                                {
                                    "key": l.get("key", "undefined"),
                                    "summary": l.get("summary", "undefined"),
                                    "description": redact_sensitive_data(l.get("description", "No description"))
                                }
                                for l in issue_data.get("linked_issues", [])
                            ]
            },
            "heatmapData": heatmap_data
        })    


    except Exception as e:
        return jsonify({"testCases": f"Error: {str(e)}"}), 500


# Main Call
if __name__ == '__main__':
    app.run(debug=True)
