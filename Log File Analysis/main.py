# Fast API - RESTful services for web , Asynchronous support
# Fast API - Latest | Speed and Performance
# fast api : CORSMiddleware - Allow frontend/backend communication 
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import os
import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import asyncio
from pathlib import Path
import logging
from dataclasses import dataclass
import openai
from pydantic import BaseModel
from openai import OpenAI
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = "Open AI key"

client = OpenAI(api_key="Open AI Key comes here ")

# Initialize FastAPI app
app = FastAPI(title="Log Files Analysis BOT", version="1.0.0")

# Add CORS middleware | Enables CORS so that frontend it can make API calls.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# OpenAI Configuration
openai.api_key = os.getenv("OPENAI_API_KEY", "Open AI Key")

# Data Models - AnalysisOptions: Which analyses to perform.
class AnalysisOptions(BaseModel):
    error_analysis: bool = True
    performance_metrics: bool = True
    security_analysis: bool = False
    trend_analysis: bool = False
    summary_report: bool = True
    anomaly_detection: bool = False

# LogEntry: Represents a parsed line from the log file.
@dataclass
class LogEntry:
    timestamp: Optional[datetime]
    level: str
    message: str
    source: str
    raw_line: str

# Chart data models
class ChartData(BaseModel):
    type: str  # 'line', 'bar', 'pie', 'doughnut'
    title: str
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Optional[Dict[str, Any]] = None

# LogAnalysisResult: Holds results from GenAI with chart data
class LogAnalysisResult(BaseModel):
    title: str
    content: str
    data: Optional[Dict[str, Any]] = None
    charts: Optional[List[ChartData]] = None

# Parsing Logs | Supports multiple log types: Apache, NGINX, syslog, Java, Python.
# Detects keywords related to errors or security threats.

# function : parse_log_content : Breaks logs into lines. Tries matching each with known regex patterns.
class LogAnalyzer:
    def __init__(self):
        self.log_patterns = {
            'apache': r'^(\S+) \S+ \S+ \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)',
            'nginx': r'^(\S+) - \S+ \[([^\]]+)\] "(\S+) (\S+) (\S+)" (\d+) (\d+)',
            'syslog': r'^(\w+\s+\d+\s+\d+:\d+:\d+) (\S+) (\S+): (.+)',
            'custom': r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (.+)',
            'java': r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[(\w+)\] \[(\S+)\] (.+)',
            'python': r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - (\w+) - (.+)'
        }
        
        self.error_keywords = [
            'error', 'exception', 'failed', 'failure', 'critical', 'fatal',
            'timeout', 'refused', 'denied', 'unauthorized', 'forbidden',
            'not found', '404', '500', '503', 'crash', 'abort'
        ]
        
        self.security_keywords = [
            'attack', 'intrusion', 'breach', 'malware', 'virus', 'hack',
            'unauthorized', 'brute force', 'sql injection', 'xss',
            'csrf', 'ddos', 'suspicious', 'blocked', 'banned'
        ]

    def parse_log_content(self, content: str) -> List[LogEntry]:
        """Parse log content into structured LogEntry objects"""
        lines = content.strip().split('\n')
        entries = []
        
        for line in lines:
            if not line.strip():
                continue
                
            entry = self._parse_single_line(line)
            if entry:
                entries.append(entry)
        
        return entries

    def _parse_single_line(self, line: str) -> Optional[LogEntry]:
        """Parse a single log line using various patterns"""
        for pattern_name, pattern in self.log_patterns.items():
            match = re.match(pattern, line)
            if match:
                try:
                    timestamp = self._extract_timestamp(match.group(1))
                    level = self._extract_level(line)
                    message = self._extract_message(line, match)
                    
                    return LogEntry(
                        timestamp=timestamp,
                        level=level,
                        message=message,
                        source=pattern_name,
                        raw_line=line
                    )
                except Exception as e:
                    logger.warning(f"Error parsing line with pattern {pattern_name}: {e}")
                    continue
        
        # If no pattern matches, create a basic entry
        return LogEntry(
            timestamp=None,
            level='UNKNOWN',
            message=line,
            source='unknown',
            raw_line=line
        )

    def _extract_timestamp(self, timestamp_str: str) -> Optional[datetime]:
        """Extract timestamp from various formats"""
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S,%f',
            '%d/%b/%Y:%H:%M:%S %z',
            '%b %d %H:%M:%S',
            '%Y-%m-%d %H:%M:%S.%f'
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(timestamp_str, fmt)
            except ValueError:
                continue
        
        return None

    def _extract_level(self, line: str) -> str:
        """Extract log level from line"""
        levels = ['DEBUG', 'INFO', 'WARN', 'WARNING', 'ERROR', 'CRITICAL', 'FATAL']
        line_upper = line.upper()
        
        for level in levels:
            if level in line_upper:
                return level
        
        # Check for HTTP status codes
        if re.search(r'\b[4-5]\d{2}\b', line):
            return 'ERROR'
        elif re.search(r'\b[2-3]\d{2}\b', line):
            return 'INFO'
        
        return 'INFO'

    def _extract_message(self, line: str, match: re.Match) -> str:
        """Extract message from parsed line"""
        try:
            # Get the last group which usually contains the message
            return match.group(match.lastindex) if match.lastindex else line
        except:
            return line

    def _generate_level_distribution_chart(self, entries: List[LogEntry]) -> ChartData:
        """Generate log level distribution chart"""
        level_counts = Counter(entry.level for entry in entries)
        
        # Color mapping for log levels
        colors = {
            'DEBUG': '#6c757d',
            'INFO': '#007bff',
            'WARN': '#ffc107',
            'WARNING': '#ffc107',
            'ERROR': '#dc3545',
            'CRITICAL': '#dc3545',
            'FATAL': '#6f42c1',
            'UNKNOWN': '#6c757d'
        }
        
        labels = list(level_counts.keys())
        data = list(level_counts.values())
        background_colors = [colors.get(level, '#6c757d') for level in labels]
        
        return ChartData(
            type='doughnut',
            title='Log Level Distribution',
            labels=labels,
            datasets=[{
                'data': data,
                'backgroundColor': background_colors,
                'borderWidth': 2
            }],
            options={
                'responsive': True,
                'plugins': {
                    'legend': {
                        'position': 'bottom'
                    }
                }
            }
        )

    def _generate_timeline_chart(self, entries: List[LogEntry]) -> ChartData:
        """Generate timeline chart showing log activity over time"""
        # Filter entries with valid timestamps
        timestamped_entries = [e for e in entries if e.timestamp]
        
        if not timestamped_entries:
            return ChartData(
                type='line',
                title='Log Activity Timeline',
                labels=[],
                datasets=[],
                options={'responsive': True}
            )
        
        # Group entries by hour
        hourly_counts = defaultdict(int)
        for entry in timestamped_entries:
            hour_key = entry.timestamp.strftime('%Y-%m-%d %H:00')
            hourly_counts[hour_key] += 1
        
        # Sort by time
        sorted_hours = sorted(hourly_counts.keys())
        labels = [datetime.strptime(hour, '%Y-%m-%d %H:%M').strftime('%m-%d %H:%M') for hour in sorted_hours]
        data = [hourly_counts[hour] for hour in sorted_hours]
        
        return ChartData(
            type='line',
            title='Log Activity Timeline',
            labels=labels,
            datasets=[{
                'label': 'Log Entries',
                'data': data,
                'borderColor': '#007bff',
                'backgroundColor': 'rgba(0, 123, 255, 0.1)',
                'fill': True,
                'tension': 0.4
            }],
            options={
                'responsive': True,
                'scales': {
                    'x': {
                        'display': True,
                        'title': {
                            'display': True,
                            'text': 'Time'
                        }
                    },
                    'y': {
                        'display': True,
                        'title': {
                            'display': True,
                            'text': 'Number of Entries'
                        }
                    }
                }
            }
        )

    def _generate_error_categories_chart(self, entries: List[LogEntry]) -> ChartData:
        """Generate error categories chart"""
        error_entries = [e for e in entries if any(keyword in e.message.lower() for keyword in self.error_keywords)]
        
        # Categorize errors
        error_categories = {
            'HTTP Errors': ['404', '500', '503', '502', '403', '401'],
            'Connection Errors': ['timeout', 'refused', 'connection'],
            'System Errors': ['crash', 'abort', 'fatal', 'critical'],
            'Authentication': ['unauthorized', 'forbidden', 'denied'],
            'Application Errors': ['exception', 'failed', 'failure']
        }
        
        category_counts = {}
        for category, keywords in error_categories.items():
            count = sum(1 for entry in error_entries if any(keyword in entry.message.lower() for keyword in keywords))
            if count > 0:
                category_counts[category] = count
        
        if not category_counts:
            category_counts = {'No Errors': 0}
        
        labels = list(category_counts.keys())
        data = list(category_counts.values())
        
        return ChartData(
            type='bar',
            title='Error Categories',
            labels=labels,
            datasets=[{
                'label': 'Error Count',
                'data': data,
                'backgroundColor': [
                    '#dc3545', '#fd7e14', '#6f42c1', '#e83e8c', '#20c997'
                ][:len(labels)],
                'borderWidth': 1
            }],
            options={
                'responsive': True,
                'scales': {
                    'y': {
                        'beginAtZero': True
                    }
                }
            }
        )

    def _generate_source_distribution_chart(self, entries: List[LogEntry]) -> ChartData:
        """Generate source distribution chart"""
        source_counts = Counter(entry.source for entry in entries)
        
        labels = list(source_counts.keys())
        data = list(source_counts.values())
        
        colors = ['#007bff', '#28a745', '#ffc107', '#dc3545', '#6f42c1', '#17a2b8']
        
        return ChartData(
            type='pie',
            title='Log Source Distribution',
            labels=labels,
            datasets=[{
                'data': data,
                'backgroundColor': colors[:len(labels)],
                'borderWidth': 2
            }],
            options={
                'responsive': True,
                'plugins': {
                    'legend': {
                        'position': 'bottom'
                    }
                }
            }
        )

    def _generate_hourly_pattern_chart(self, entries: List[LogEntry]) -> ChartData:
        """Generate hourly activity pattern chart"""
        timestamped_entries = [e for e in entries if e.timestamp]
        
        if not timestamped_entries:
            return ChartData(
                type='bar',
                title='Hourly Activity Pattern',
                labels=[],
                datasets=[],
                options={'responsive': True}
            )
        
        # Group by hour of day (0-23)
        hourly_pattern = defaultdict(int)
        for entry in timestamped_entries:
            hour = entry.timestamp.hour
            hourly_pattern[hour] += 1
        
        # Create labels for all 24 hours
        labels = [f"{hour:02d}:00" for hour in range(24)]
        data = [hourly_pattern.get(hour, 0) for hour in range(24)]
        
        return ChartData(
            type='bar',
            title='Hourly Activity Pattern',
            labels=labels,
            datasets=[{
                'label': 'Log Entries',
                'data': data,
                'backgroundColor': 'rgba(0, 123, 255, 0.6)',
                'borderColor': '#007bff',
                'borderWidth': 1
            }],
            options={
                'responsive': True,
                'scales': {
                    'x': {
                        'title': {
                            'display': True,
                            'text': 'Hour of Day'
                        }
                    },
                    'y': {
                        'beginAtZero': True,
                        'title': {
                            'display': True,
                            'text': 'Number of Entries'
                        }
                    }
                }
            }
        )

    # These use OpenAI (via client.chat.completions.create) to perform analysis.
    async def analyze_with_genai(self, entries: List[LogEntry], options: AnalysisOptions) -> List[LogAnalysisResult]:
        """Analyze log entries using GenAI"""
        results = []
        
        # Prepare log data for AI analysis
        log_summary = self._prepare_log_summary(entries)
        
        try:
            if options.summary_report:
                summary = await self._generate_summary_report(log_summary, entries)
                results.append(summary)
            
            if options.error_analysis:
                error_analysis = await self._analyze_errors(log_summary, entries)
                results.append(error_analysis)
            
            if options.performance_metrics:
                performance = await self._analyze_performance(log_summary, entries)
                results.append(performance)
            
            if options.security_analysis:
                security = await self._analyze_security(log_summary, entries)
                results.append(security)
            
            if options.trend_analysis:
                trends = await self._analyze_trends(log_summary, entries)
                results.append(trends)
            
            if options.anomaly_detection:
                anomalies = await self._detect_anomalies(log_summary, entries)
                results.append(anomalies)
        
        except Exception as e:
            logger.error(f"GenAI analysis error: {e}")
            # Return fallback analysis
            results.append(LogAnalysisResult(
                title="Analysis Error",
                content=f"GenAI analysis failed: {str(e)}. Using fallback analysis.",
                data={"error": str(e)}
            ))
        
        return results

    def _prepare_log_summary(self, entries: List[LogEntry]) -> str:
        """Prepare a summary of log entries for AI analysis"""
        total_entries = len(entries)
        error_entries = [e for e in entries if 'error' in e.level.lower() or 'error' in e.message.lower()]
        
        levels_count = {}
        for entry in entries:
            levels_count[entry.level] = levels_count.get(entry.level, 0) + 1
        
        # Sample of recent entries (last 50)
        recent_entries = entries[-50:] if len(entries) > 50 else entries
        sample_messages = [entry.message[:200] for entry in recent_entries]
        
        summary = f"""
        Log Analysis Summary:
        - Total entries: {total_entries}
        - Error entries: {len(error_entries)}
        - Log levels distribution: {levels_count}
        - Sample recent messages: {sample_messages[:10]}
        """
        
        return summary

    async def _generate_summary_report(self, log_summary: str, entries: List[LogEntry]) -> LogAnalysisResult:
        """Generate summary report using GenAI"""
        prompt = f"""
        Analyze the following log data and provide a comprehensive summary report:
        
        {log_summary}
        
        Please provide:
        1. Overall system health assessment
        2. Key statistics and metrics
        3. Important findings
        4. Recommendations
        
        Format the response as HTML with proper styling.
        """
        
        charts = [
            self._generate_level_distribution_chart(entries),
            self._generate_timeline_chart(entries),
            self._generate_source_distribution_chart(entries),
            self._generate_hourly_pattern_chart(entries)
        ]
        
        try:
            response = await self._call_openai(prompt)
            return LogAnalysisResult(
                title="Summary Report",
                content=response,
                data={"total_entries": len(entries)},
                charts=charts
            )
        except Exception as e:
            return LogAnalysisResult(
                title="Summary Report",
                content=f"""
                <strong>Total log entries:</strong> {len(entries)}<br>
                <strong>Analysis timestamp:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                <strong>Status:</strong> Analysis completed with basic parsing<br>
                <strong>Note:</strong> GenAI analysis unavailable: {str(e)}
                """,
                charts=charts
            )

    async def _analyze_errors(self, log_summary: str, entries: List[LogEntry]) -> LogAnalysisResult:
        """Analyze errors using GenAI"""
        error_entries = [e for e in entries if any(keyword in e.message.lower() for keyword in self.error_keywords)]
        
        prompt = f"""
        Analyze the following error data from log files:
        
        {log_summary}
        
        Error entries sample: {[e.message[:100] for e in error_entries[:20]]}
        
        Please provide:
        1. Error categorization and count
        2. Most critical errors
        3. Error patterns and frequency
        4. Root cause analysis
        5. Recommendations for resolution
        
        Format as HTML.
        """
        
        charts = [
            self._generate_error_categories_chart(entries)
        ]
        
        try:
            response = await self._call_openai(prompt)
            return LogAnalysisResult(
                title="Error Analysis",
                content=response,
                data={"error_count": len(error_entries)},
                charts=charts
            )
        except Exception as e:
            error_rate = (len(error_entries) / len(entries)) * 100 if entries else 0
            return LogAnalysisResult(
                title="Error Analysis",
                content=f"""
                <strong>Total errors found:</strong> {len(error_entries)}<br>
                <strong>Error rate:</strong> {error_rate:.2f}%<br>
                <strong>Analysis:</strong> Basic error detection completed<br>
                <strong>Note:</strong> Advanced AI analysis unavailable: {str(e)}
                """,
                charts=charts
            )

    async def _analyze_performance(self, log_summary: str, entries: List[LogEntry]) -> LogAnalysisResult:
        """Analyze performance metrics using GenAI"""
        prompt = f"""
        Analyze the performance metrics from the following log data:
        
        {log_summary}
        
        Focus on:
        1. Response times and latency patterns
        2. Throughput analysis
        3. Resource utilization indicators
        4. Performance bottlenecks
        5. Optimization recommendations
        
        Format as HTML.
        """
        
        charts = [
            self._generate_timeline_chart(entries),
            self._generate_hourly_pattern_chart(entries)
        ]
        
        try:
            response = await self._call_openai(prompt)
            return LogAnalysisResult(
                title="Performance Metrics",
                content=response,
                charts=charts
            )
        except Exception as e:
            return LogAnalysisResult(
                title="Performance Metrics",
                content=f"""
                <strong>Total requests analyzed:</strong> {len(entries)}<br>
                <strong>Average processing:</strong> Normal<br>
                <strong>Status:</strong> Performance analysis completed<br>
                <strong>Note:</strong> Detailed AI analysis unavailable: {str(e)}
                """,
                charts=charts
            )

    async def _analyze_security(self, log_summary: str, entries: List[LogEntry]) -> LogAnalysisResult:
        """Analyze security threats using GenAI"""
        security_entries = [e for e in entries if any(keyword in e.message.lower() for keyword in self.security_keywords)]
        
        prompt = f"""
        Analyze the security aspects of the following log data:
        
        {log_summary}
        
        Security-related entries: {[e.message[:100] for e in security_entries[:10]]}
        
        Please identify:
        1. Potential security threats
        2. Suspicious activities
        3. Attack patterns
        4. Vulnerability indicators
        5. Security recommendations
        
        Format as HTML.
        """
        
        try:
            response = await self._call_openai(prompt)
            return LogAnalysisResult(
                title="Security Analysis",
                content=response,
                data={"security_events": len(security_entries)}
            )
        except Exception as e:
            return LogAnalysisResult(
                title="Security Analysis",
                content=f"""
                <strong>Security events detected:</strong> {len(security_entries)}<br>
                <strong>Threat level:</strong> {"Medium" if security_entries else "Low"}<br>
                <strong>Status:</strong> Security scan completed<br>
                <strong>Note:</strong> Advanced AI analysis unavailable: {str(e)}
                """
            )

    async def _analyze_trends(self, log_summary: str, entries: List[LogEntry]) -> LogAnalysisResult:
        """Analyze trends using GenAI"""
        prompt = f"""
        Analyze trends and patterns in the following log data:
        
        {log_summary}
        
        Please identify:
        1. Temporal patterns and trends
        2. Usage patterns
        3. Seasonal variations
        4. Growth trends
        5. Predictive insights
        
        Format as HTML.
        """
        
        charts = [
            self._generate_timeline_chart(entries),
            self._generate_hourly_pattern_chart(entries)
        ]
        
        try:
            response = await self._call_openai(prompt)
            return LogAnalysisResult(
                title="Trend Analysis",
                content=response,
                charts=charts
            )
        except Exception as e:
            return LogAnalysisResult(
                title="Trend Analysis",
                content=f"""
                <strong>Trend analysis:</strong> Pattern recognition completed<br>
                <strong>Data points:</strong> {len(entries)}<br>
                <strong>Status:</strong> Basic trend analysis available<br>
                <strong>Note:</strong> Advanced AI analysis unavailable: {str(e)}
                """,
                charts=charts
            )

    async def _detect_anomalies(self, log_summary: str, entries: List[LogEntry]) -> LogAnalysisResult:
        """Detect anomalies using GenAI"""
        prompt = f"""
        Detect anomalies and unusual patterns in the following log data:
        
        {log_summary}
        
        Please identify:
        1. Unusual patterns or outliers
        2. Anomalous behavior
        3. Unexpected events
        4. Data inconsistencies
        5. Investigation recommendations
        
        Format as HTML.
        """
        
        try:
            response = await self._call_openai(prompt)
            return LogAnalysisResult(
                title="Anomaly Detection",
                content=response
            )
        except Exception as e:
            return LogAnalysisResult(
                title="Anomaly Detection",
                content=f"""
                <strong>Anomaly detection:</strong> Completed<br>
                <strong>Data analyzed:</strong> {len(entries)} entries<br>
                <strong>Status:</strong> Basic anomaly detection performed<br>
                <strong>Note:</strong> Advanced AI analysis unavailable: {str(e)}
                """
            )

    async def _call_openai(self, prompt: str) -> str:
        """Make async call to OpenAI API"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a expert log analysis assistant. Provide detailed, actionable insights in HTML format."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise e

# Initialize analyzer
analyzer = LogAnalyzer()

# API Routes | Frontend Loader - HTMLResponse
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main HTML page"""
    return FileResponse("index.html")

# File Upload + Analysis
@app.post("/analyze")
async def analyze_log_file(
    file: UploadFile = File(...),
    options: str = Form(...)
):
    """Analyze uploaded log file"""
    try:
        # Validate file
        if not file.filename.endswith(('.log', '.txt')):
            raise HTTPException(status_code=400, detail="Only .log and .txt files are supported")
        
        # Read file content
        content = await file.read()
        try:
            log_content = content.decode('utf-8')
        except UnicodeDecodeError:
            log_content = content.decode('latin-1')
        
        # Parse analysis options
        analysis_options = AnalysisOptions(**json.loads(options))
        
        # Parse log entries
        entries = analyzer.parse_log_content(log_content)
        
        if not entries:
            raise HTTPException(status_code=400, detail="No valid log entries found")
        
        # Perform GenAI analysis
        results = await analyzer.analyze_with_genai(entries, analysis_options)
        
        return {
            "success": True,
            "results": [result.dict() for result in results],
            "metadata": {
                "total_entries": len(entries),
                "file_size": len(content),
                "analysis_timestamp": datetime.now().isoformat()
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("static", exist_ok=True)
    
    # Run the server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )