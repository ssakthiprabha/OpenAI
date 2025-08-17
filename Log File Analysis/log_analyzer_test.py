import re
import json
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
import openai
from typing import Dict, List, Any

class LogAnalysisBot:
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the Log Analysis Bot with OpenAI API key
        """
        self.openai_client = None
        if openai_api_key and openai_api_key != 'your-openai-api-key-here':
            try:
                openai.api_key = openai_api_key
                self.openai_client = openai
            except Exception as e:
                print(f"OpenAI initialization failed: {e}")
        
        # Common log patterns
        self.patterns = {
            'apache': r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<url>[^"]*)" (?P<status>\d+) (?P<size>\d+)',
            'nginx': r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<timestamp>[^\]]+)\] "(?P<method>\w+) (?P<url>[^"]*)" (?P<status>\d+) (?P<size>\d+)',
            'error': r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(?P<level>\w+)\] (?P<message>.*)',
            'application': r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[\.\d]*) (?P<level>\w+) (?P<logger>[^\s]+) - (?P<message>.*)',
            'syslog': r'(?P<timestamp>\w{3} \d{1,2} \d{2}:\d{2}:\d{2}) (?P<hostname>\w+) (?P<process>[^:]+): (?P<message>.*)',
            'json': r'^\{.*\}$',
            'iis': r'(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (?P<ip>\d+\.\d+\.\d+\.\d+) (?P<method>\w+) (?P<url>[^\s]+) (?P<status>\d+)',
            'django': r'\[(?P<timestamp>[^\]]+)\] (?P<level>\w+) (?P<message>.*)',
            'rails': r'(?P<level>\w+), \[(?P<timestamp>[^\]]+)\] (?P<message>.*)'
        }
        
        self.parsed_logs = []
        self.log_stats = {}
        
    def detect_log_format(self, sample_lines: List[str]) -> str:
        """
        Detect the log format based on sample lines
        """
        format_scores = {}
        
        for format_name, pattern in self.patterns.items():
            matches = 0
            for line in sample_lines[:20]:  # Test first 20 lines
                line = line.strip()
                if not line:
                    continue
                try:
                    if format_name == 'json':
                        json.loads(line)
                        matches += 1
                    elif re.match(pattern, line):
                        matches += 1
                except:
                    continue
            format_scores[format_name] = matches
        
        if format_scores:
            detected_format = max(format_scores, key=format_scores.get)
            return detected_format if format_scores[detected_format] > 0 else 'unknown'
        return 'unknown'
    
    def parse_log_file(self, file_path: str, log_format: str = None) -> Dict:
    """
    Parse log file and extract structured data
    """
    try:
        # Read file and close immediately
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        if not log_format:
            log_format = self.detect_log_format(lines)

        parsed_data = []
        errors = []

        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                if log_format == 'json':
                    data = json.loads(line)
                    data['line_number'] = line_num
                    data['raw_line'] = line
                    data['log_format'] = log_format
                    parsed_data.append(data)
                else:
                    pattern = self.patterns.get(log_format, self.patterns['application'])
                    match = re.match(pattern, line)
                    if match:
                        data = match.groupdict()
                        data['line_number'] = line_num
                        data['raw_line'] = line
                        data['log_format'] = log_format
                        parsed_data.append(data)
                    else:
                        parsed_data.append({
                            'line_number': line_num,
                            'raw_line': line,
                            'log_format': 'unparsed',
                            'message': line
                        })
            except Exception as e:
                errors.append(f"Line {line_num}: {str(e)}")
                if len(errors) >= 50:
                    break

        self.parsed_logs = parsed_data

        return {
            'success': True,
            'parsed_count': len(parsed_data),
            'total_lines': len(lines),
            'detected_format': log_format,
            'errors': errors[:10]
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'parsed_count': 0,
            'total_lines': 0
        }
    
    def analyze_basic_stats(self) -> Dict:
        """
        Generate basic statistics from parsed logs
        """
        if not self.parsed_logs:
            return {}
        
        df = pd.DataFrame(self.parsed_logs)
        
        stats = {
            'total_lines': len(self.parsed_logs),
            'log_formats': df['log_format'].value_counts().to_dict(),
            'date_range': {},
            'status_codes': {},
            'ip_addresses': {},
            'log_levels': {},
            'error_patterns': [],
            'url_patterns': {},
            'methods': {},
            'user_agents': {},
            'response_sizes': {}
        }
        
        # Analyze timestamps
        if 'timestamp' in df.columns:
            timestamps = df['timestamp'].dropna()
            if len(timestamps) > 0:
                stats['date_range'] = {
                    'first_entry': str(timestamps.iloc[0]),
                    'last_entry': str(timestamps.iloc[-1]),
                    'total_entries': len(timestamps)
                }
        
        # Analyze status codes
        if 'status' in df.columns:
            stats['status_codes'] = df['status'].value_counts().head(15).to_dict()
        
        # Analyze IP addresses
        if 'ip' in df.columns:
            stats['ip_addresses'] = df['ip'].value_counts().head(15).to_dict()
        
        # Analyze log levels
        if 'level' in df.columns:
            stats['log_levels'] = df['level'].value_counts().to_dict()
        
        # Analyze URLs
        if 'url' in df.columns:
            stats['url_patterns'] = df['url'].value_counts().head(15).to_dict()
        
        # Analyze HTTP methods
        if 'method' in df.columns:
            stats['methods'] = df['method'].value_counts().to_dict()
        
        # Analyze response sizes
        if 'size' in df.columns:
            try:
                sizes = pd.to_numeric(df['size'], errors='coerce').dropna()
                if len(sizes) > 0:
                    stats['response_sizes'] = {
                        'min': int(sizes.min()),
                        'max': int(sizes.max()),
                        'avg': int(sizes.mean()),
                        'total': int(sizes.sum())
                    }
            except:
                pass
        
        # Find error patterns
        error_keywords = ['error', 'exception', 'fail', 'critical', 'fatal', 'warning', 'denied', 'timeout', 'refused']
        error_lines = []
        
        for log in self.parsed_logs:
            message = log.get('message', log.get('raw_line', '')).lower()
            level = log.get('level', '').lower()
            
            # Check for error keywords or error levels
            if (any(keyword in message for keyword in error_keywords) or 
                level in ['error', 'critical', 'fatal', 'severe', 'warn', 'warning']):
                error_lines.append({
                    'line_number': log['line_number'],
                    'message': log.get('message', log.get('raw_line', ''))[:300],
                    'level': log.get('level', 'unknown'),
                    'timestamp': log.get('timestamp', 'unknown')
                })
        
        stats['error_patterns'] = error_lines[:25]  # Top 25 errors
        
        self.log_stats = stats
        return stats
    
    def find_anomalies(self) -> List[Dict]:
        """
        Find anomalies in log data
        """
        anomalies = []
        
        if not self.parsed_logs:
            return anomalies
        
        df = pd.DataFrame(self.parsed_logs)
        
        # HTTP error status codes
        if 'status' in df.columns:
            try:
                status_codes = df['status'].astype(str)
                error_4xx = status_codes.str.match(r'^4\d\d$').sum()
                error_5xx = status_codes.str.match(r'^5\d\d$').sum()
                total_requests = len(df)
                
                if error_4xx > 0:
                    error_rate = (error_4xx / total_requests) * 100
                    anomalies.append({
                        'type': 'HTTP_4XX_ERRORS',
                        'severity': 'high' if error_rate > 20 else 'medium' if error_rate > 10 else 'low',
                        'count': error_4xx,
                        'description': f"Found {error_4xx} client errors (4xx) - {error_rate:.1f}% of requests",
                        'top_errors': status_codes[status_codes.str.match(r'^4\d\d$')].value_counts().head(5).to_dict()
                    })
                
                if error_5xx > 0:
                    error_rate = (error_5xx / total_requests) * 100
                    anomalies.append({
                        'type': 'HTTP_5XX_ERRORS',
                        'severity': 'high' if error_rate > 5 else 'medium' if error_rate > 1 else 'low',
                        'count': error_5xx,
                        'description': f"Found {error_5xx} server errors (5xx) - {error_rate:.1f}% of requests",
                        'top_errors': status_codes[status_codes.str.match(r'^5\d\d$')].value_counts().head(5).to_dict()
                    })
            except:
                pass
        
        # Suspicious IP patterns
        if 'ip' in df.columns:
            try:
                ip_counts = df['ip'].value_counts()
                mean_requests = ip_counts.mean()
                std_requests = ip_counts.std()
                threshold = mean_requests + (2 * std_requests)  # 2 standard deviations
                
                suspicious_ips = ip_counts[ip_counts > max(threshold, 100)]
                if len(suspicious_ips) > 0:
                    anomalies.append({
                        'type': 'SUSPICIOUS_IP_ACTIVITY',
                        'severity': 'high' if suspicious_ips.iloc[0] > 1000 else 'medium',
                        'count': len(suspicious_ips),
                        'description': f"Found {len(suspicious_ips)} IPs with unusually high activity (>{threshold:.0f} requests)",
                        'top_ips': suspicious_ips.head(5).to_dict()
                    })
            except:
                pass
        
        # Error level analysis
        if 'level' in df.columns:
            try:
                error_levels = ['ERROR', 'CRITICAL', 'FATAL', 'SEVERE']
                error_logs = df[df['level'].str.upper().isin(error_levels)]
                total_logs = len(df)
                
                if len(error_logs) > 0:
                    error_rate = (len(error_logs) / total_logs) * 100
                    anomalies.append({
                        'type': 'HIGH_ERROR_RATE',
                        'severity': 'high' if error_rate > 20 else 'medium' if error_rate > 10 else 'low',
                        'count': len(error_logs),
                        'description': f"High error rate: {len(error_logs)} errors out of {total_logs} logs ({error_rate:.1f}%)",
                        'error_distribution': error_logs['level'].value_counts().to_dict()
                    })
            except:
                pass
        
        # Check for potential security issues
        security_patterns = [
            (r'sql.*injection', 'SQL_INJECTION_ATTEMPT'),
            (r'<script.*>', 'XSS_ATTEMPT'),
            (r'\.\./', 'DIRECTORY_TRAVERSAL'),
            (r'union.*select', 'SQL_INJECTION_ATTEMPT'),
            (r'exec.*xp_', 'SQL_INJECTION_ATTEMPT'),
            (r'/etc/passwd', 'FILE_INCLUSION_ATTEMPT'),
            (r'cmd\.exe', 'COMMAND_INJECTION')
        ]
        
        for pattern, threat_type in security_patterns:
            try:
                matches = []
                for log in self.parsed_logs:
                    content = log.get('message', log.get('raw_line', '')).lower()
                    if re.search(pattern, content, re.IGNORECASE):
                        matches.append(log)
                
                if matches:
                    anomalies.append({
                        'type': threat_type,
                        'severity': 'high',
                        'count': len(matches),
                        'description': f"Potential {threat_type.replace('_', ' ').lower()} detected in {len(matches)} log entries",
                        'sample_ips': list(set([log.get('ip', 'unknown') for log in matches[:5]]))
                    })
            except:
                continue
        
        return anomalies
    
    def generate_ai_insights(self, stats: Dict, anomalies: List[Dict]) -> str:
        """
        Generate AI-powered insights using OpenAI
        """
        if not self.openai_client:
            return "AI insights not available. Please set OPENAI_API_KEY environment variable to enable AI-powered analysis."
        
        try:
            # Create a concise summary for AI analysis
            summary = {
                'total_logs': stats.get('total_lines', 0),
                'formats': list(stats.get('log_formats', {}).keys()),
                'error_count': len(stats.get('error_patterns', [])),
                'status_codes': dict(list(stats.get('status_codes', {}).items())[:5]),
                'anomalies_count': len(anomalies),
                'anomaly_types': [a['type'] for a in anomalies],
                'high_severity_count': len([a for a in anomalies if a.get('severity') == 'high'])
            }
            
            prompt = f"""
            Analyze this log file data and provide insights:
            
            Log Summary:
            - Total logs: {summary['total_logs']}
            - Formats detected: {', '.join(summary['formats'])}
            - Error entries: {summary['error_count']}
            - Anomalies found: {summary['anomalies_count']} ({summary['high_severity_count']} high severity)
            
            Anomaly Types: {', '.join(summary['anomaly_types'])}
            
            Top Status Codes: {summary['status_codes']}
            
            Please provide:
            1. Overall system health assessment (1-2 sentences)
            2. Key security concerns (if any)
            3. Performance issues identified
            4. 2-3 actionable recommendations
            
            Keep response concise but informative (under 250 words).
            """
            
            response = self.openai_client.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a log analysis expert. Provide concise, actionable insights about system health, security, and performance."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=350,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"AI analysis failed: {str(e)}. Please check your OpenAI API key and try again."
    
    def generate_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive analysis report
        """
        stats = self.analyze_basic_stats()
        anomalies = self.find_anomalies()
        ai_insights = self.generate_ai_insights(stats, anomalies)
        
        report = f"""
LOG ANALYSIS REPORT
===================
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY
-------
Total Log Lines: {stats.get('total_lines', 0):,}
Detected Formats: {', '.join(stats.get('log_formats', {}).keys())}
Error Patterns Found: {len(stats.get('error_patterns', []))}
Anomalies Detected: {len(anomalies)}

STATISTICS
----------
{json.dumps(stats, indent=2, default=str)}

ANOMALIES
---------
{json.dumps(anomalies, indent=2, default=str)}

AI INSIGHTS
-----------
{ai_insights}
"""
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            print(f"Report saved to {output_file}")
        
        return report