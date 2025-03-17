from flask import Flask, request, render_template, jsonify
import pandas as pd
import openai
import os
import matplotlib.pyplot as plt
import seaborn as sns  # Import Seaborn for advanced visualization
import time  # For unique filenames
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend

# Set OpenAI API Key
openai.api_key = 'your OpenAI API key'  # Replace with your key

app = Flask(__name__)
UPLOAD_FOLDER = "/Users/ssathiyaseelan/Desktop/python/POC_22_07/Document_Search_BOT_l2/upload"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CHART_FOLDER = "/Users/ssathiyaseelan/Desktop/python/POC_22_07/Document_Search_BOT_l2/static/charts"
os.makedirs(CHART_FOLDER, exist_ok=True)  # Ensure chart folder exists

uploaded_data = None  # Store uploaded data globally

def load_excel(file_path):
    """Reads the Excel file and returns a dictionary of DataFrames."""
    try:
        xls = pd.ExcelFile(file_path)
        return {sheet_name: xls.parse(sheet_name) for sheet_name in xls.sheet_names}
    except Exception as e:
        return str(e)

def generate_chart(df, chart_type, x_col=None, y_col=None):
    """Generates a chart and saves it as an image."""
    plt.figure(figsize=(8, 5))
    
    if chart_type == "bar":
        df_grouped = df.groupby(x_col, as_index=False)[y_col].sum()
        plt.bar(df_grouped[x_col], df_grouped[y_col], color="skyblue")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"{y_col} by {x_col}")
    
    elif chart_type == "pie":
        df_grouped = df.groupby(x_col, as_index=False)[y_col].sum()
        plt.pie(df_grouped[y_col], labels=df_grouped[x_col], autopct="%1.1f%%", colors=plt.cm.Paired.colors)
        plt.title(f"{y_col} Distribution")
    
    elif chart_type == "line":
        df_grouped = df.groupby(x_col, as_index=False)[y_col].sum()
        plt.plot(df_grouped[x_col], df_grouped[y_col], marker="o", linestyle="-", color="blue")
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(f"Trend of {y_col} by {x_col}")
        plt.grid()
    
    elif chart_type == "heatmap":
        plt.figure(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=['number'])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title("Heatmap of Correlation Matrix")
    
    elif chart_type == "correlation":
        plt.figure(figsize=(8, 6))
        numeric_df = df.select_dtypes(include=['number'])
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap="viridis", linewidths=0.5)
        plt.title("Correlation Matrix")
    
    filename = f"chart_{int(time.time())}.png"
    filepath = os.path.join(CHART_FOLDER, filename)
    plt.savefig(filepath)
    plt.close()
    return filename

@app.route("/", methods=["GET", "POST"])
def index():
    global uploaded_data
    chart_files = os.listdir(CHART_FOLDER) if os.path.exists(CHART_FOLDER) else []
    if request.method == "POST":
        file = request.files["file"]
        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            uploaded_data = load_excel(file_path)
    return render_template("index.html", sheets=uploaded_data.keys() if uploaded_data else [], charts=chart_files)

@app.route("/ask", methods=["POST"])
def ask():
    global uploaded_data
    if not uploaded_data:
        return jsonify({"answer": "No file uploaded yet!"})

    user_query = request.form["question"]
    context = "\n".join([df.to_string(index=False) for df in uploaded_data.values()])
    
    prompt = f"Refer to the following Excel data:\n{context}\n\nQuestion: {user_query}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return jsonify({"answer": response["choices"][0]["message"]["content"].strip()})


@app.route("/chart", methods=["POST"])
def chart():
    global uploaded_data
    if not uploaded_data:
        return jsonify({"error": "No file uploaded yet!"})
    
    sheet_name = request.form["sheet"]
    chart_type = request.form["chartType"]
    x_col = request.form.get("xCol")
    y_col = request.form.get("yCol")

    if sheet_name not in uploaded_data:
        return jsonify({"error": "Sheet not found!"})
    
    df = uploaded_data[sheet_name]
    chart_filename = generate_chart(df, chart_type, x_col, y_col)
    
    # chart_url = f"/static/charts/{chart_filename}"
    # return jsonify({"chart_url": chart_url, "chart_name": chart_filename})
    # Get all chart files from the folder
    chart_files = sorted(os.listdir(CHART_FOLDER), reverse=True)
    chart_urls = [f"/static/charts/{file}" for file in chart_files]

    return jsonify({"chart_urls": chart_urls})

@app.route("/remove_chart", methods=["POST"])
def remove_chart():
    chart_name = request.form["chart"]
    chart_path = os.path.join(CHART_FOLDER, chart_name)

    if os.path.exists(chart_path):
        os.remove(chart_path)
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "error": "File not found"})

if __name__ == "__main__":
    app.run(debug=True)