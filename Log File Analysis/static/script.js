// static/script.js

document.addEventListener("DOMContentLoaded", function () {
    const fileInput = document.getElementById("fileInput");
    const loading = document.getElementById("loading");
    const results = document.getElementById("results");
    const statsGrid = document.getElementById("statsGrid");
    const aiInsights = document.getElementById("aiInsights");
    const anomaliesContainer = document.getElementById("anomalies");
    const errorPatterns = document.getElementById("errorPatterns");
    const detailedStats = document.getElementById("detailedStats");
    const errorModal = document.getElementById("errorModal");
    const errorMessage = document.getElementById("errorMessage");

    fileInput.addEventListener("change", function () {
        if (fileInput.files.length > 0) {
            uploadFile(fileInput.files[0]);
        }
    });

    function uploadFile(file) {
        loading.style.display = "block";
        results.style.display = "none";

        const formData = new FormData();
        formData.append("file", file);

        fetch("/upload", {
            method: "POST",
            body: formData
        })
        .then(res => {
			if (!res.ok) {
				return res.text().then(text => {
					throw new Error("Server error: " + text);
				});
			}
			return res.json();
		})
        .then(data => {
            loading.style.display = "none";
            if (data.success) {
                displayStats(data.stats);
                displayAnomalies(data.anomalies);
                displayInsights(data.ai_insights);
                displayErrorPatterns(data.stats.error_patterns);
                results.style.display = "block";
            } else {
                showError(data.error || "Unknown error occurred.");
            }
        })
        .catch(err => {
            loading.style.display = "none";
            showError("Upload failed: " + err);
        });
    }

    function displayStats(stats) {
        statsGrid.innerHTML = `
            <p><strong>Total Lines:</strong> ${stats.total_lines}</p>
            <p><strong>Log Formats:</strong> ${Object.keys(stats.log_formats).join(', ')}</p>
            <p><strong>Status Codes:</strong> ${Object.entries(stats.status_codes).map(([k,v]) => `${k}: ${v}`).join(', ')}</p>
            <p><strong>IP Addresses:</strong> ${Object.entries(stats.ip_addresses).map(([k,v]) => `${k}: ${v}`).join(', ')}</p>
            <p><strong>Log Levels:</strong> ${Object.entries(stats.log_levels).map(([k,v]) => `${k}: ${v}`).join(', ')}</p>
        `;
    }

    function displayAnomalies(anomalies) {
        anomaliesContainer.innerHTML = anomalies.map(a => `
            <div class="anomaly">
                <p><strong>Type:</strong> ${a.type}</p>
                <p><strong>Severity:</strong> ${a.severity}</p>
                <p>${a.description}</p>
            </div>
        `).join('');
    }

    function displayInsights(insight) {
        aiInsights.innerText = insight;
    }

    function displayErrorPatterns(errors) {
        errorPatterns.innerHTML = errors.map(e => `
            <div class="error-pattern">
                <p><strong>Line ${e.line_number}</strong> [${e.level || 'N/A'}]: ${e.message}</p>
            </div>
        `).join('');
    }

    function showError(message) {
        errorMessage.innerText = message;
        errorModal.style.display = "block";
    }

    document.querySelector(".modal .close").onclick = function () {
        errorModal.style.display = "none";
    };

    window.onclick = function (event) {
        if (event.target == errorModal) {
            errorModal.style.display = "none";
        }
    };
});

function resetAnalysis() {
    window.location.reload();
}

function showTab(tabName) {
    // Extend this based on tab content if needed
    detailedStats.innerHTML = `<p>Tab: <strong>${tabName}</strong> - data coming soon...</p>`;
}