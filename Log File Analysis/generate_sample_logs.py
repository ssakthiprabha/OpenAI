#!/usr/bin/env python3
"""
Sample Log File Generator for Testing the Log Analysis BOT
"""

import random
import datetime
from pathlib import Path

def generate_sample_logs():
    """Generate various types of sample log files for testing"""
    
    # Create samples directory
    samples_dir = Path("sample_logs")
    samples_dir.mkdir(exist_ok=True)
    
    # Generate Apache Access Log
    generate_apache_log(samples_dir / "apache_access.log")
    
    # Generate Application Error Log
    generate_application_log(samples_dir / "application_errors.log")
    
    # Generate System Log
    generate_system_log(samples_dir / "system.log")
    
    # Generate Security Log
    generate_security_log(samples_dir / "security.log")
    
    # Generate Performance Log
    generate_performance_log(samples_dir / "performance.log")
    
    print("✅ Sample log files generated in 'sample_logs' directory")

def generate_apache_log(filename):
    """Generate Apache-style access log"""
    
    ips = ["192.168.1.100", "10.0.0.50", "203.0.113.45", "198.51.100.10", "172.16.0.25"]
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    ]
    
    urls = [
        "/", "/index.html", "/about", "/products", "/contact",
        "/api/users", "/api/products", "/api/orders", "/admin",
        "/login", "/logout", "/register", "/dashboard", "/profile"
    ]
    
    status_codes = [200, 200, 200, 200, 200, 301, 302, 404, 500, 503]
    
    with open(filename, "w") as f:
        base_time = datetime.datetime.now() - datetime.timedelta(hours=24)
        
        for i in range(1000):
            timestamp = base_time + datetime.timedelta(minutes=i)
            ip = random.choice(ips)
            url = random.choice(urls)
            status = random.choice(status_codes)
            size = random.randint(100, 50000)
            user_agent = random.choice(user_agents)
            
            # Simulate some errors
            if random.random() < 0.1:  # 10% error rate
                status = random.choice([404, 500, 503])
            
            log_line = f'{ip} - - [{timestamp.strftime("%d/%b/%Y:%H:%M:%S %z")}] "GET {url} HTTP/1.1" {status} {size} "-" "{user_agent}"'
            f.write(log_line + "\n")

def generate_application_log(filename):
    """Generate application error log"""
    
    levels = ["INFO", "DEBUG", "WARN", "ERROR", "CRITICAL"]
    components = ["UserService", "PaymentProcessor", "DatabaseConnection", "EmailService", "CacheManager"]
    
    error_messages = [
        "Connection timeout to database",
        "Failed to process payment for user ID: {user_id}",
        "Memory usage exceeds 80% threshold",
        "Unable to send email notification",
        "Cache invalidation failed",
        "Authentication failed for user: {user}",
        "API rate limit exceeded",
        "File upload failed: invalid format",
        "Session expired for user: {user}",
        "Unable to connect to external service"
    ]
    
    with open(filename, "w") as f:
        base_time = datetime.datetime.now() - datetime.timedelta(hours=12)
        
        for i in range(500):
            timestamp = base_time + datetime.timedelta(seconds=i*30)
            level = random.choice(levels)
            component = random.choice(components)
            
            # More errors in certain components
            if component in ["DatabaseConnection", "PaymentProcessor"]:
                level = random.choice(["ERROR", "CRITICAL", "WARN"])
            
            message = random.choice(error_messages)
            if "{user_id}" in message:
                message = message.format(user_id=random.randint(1000, 9999))
            if "{user}" in message:
                message = message.format(user=f"user_{random.randint(100, 999)}")
            
            log_line = f'{timestamp.strftime("%Y-%m-%d %H:%M:%S")} [{level}] [{component}] {message}'
            f.write(log_line + "\n")

def generate_system_log(filename):
    """Generate system log"""
    
    services = ["nginx", "apache2", "mysql", "redis", "cron", "sshd", "systemd"]
    
    system_messages = [
        "Service started successfully",
        "Configuration reloaded",
        "Memory usage: {memory}%",
        "CPU usage: {cpu}%",
        "Disk usage warning: {disk}%",
        "Network interface eth0 is up",
        "Failed to start service: {service}",
        "Service stopped unexpectedly",
        "Backup completed successfully",
        "System reboot required"
    ]
    
    with open(filename, "w") as f:
        base_time = datetime.datetime.now() - datetime.timedelta(hours=6)
        
        for i in range(300):
            timestamp = base_time + datetime.timedelta(minutes=i)
            service = random.choice(services)
            
            message = random.choice(system_messages)
            if "{memory}" in message:
                message = message.format(memory=random.randint(30, 95))
            if "{cpu}" in message:
                message = message.format(cpu=random.randint(10, 85))
            if "{disk}" in message:
                message = message.format(disk=random.randint(50, 95))
            if "{service}" in message:
                message = message.format(service=random.choice(services))
            
            log_line = f'{timestamp.strftime("%b %d %H:%M:%S")} server01 {service}: {message}'
            f.write(log_line + "\n")

def generate_security_log(filename):
    """Generate security-related log"""
    
    suspicious_ips = ["192.168.1.200", "10.0.0.100", "203.0.113.99", "198.51.100.99"]
    normal_ips = ["192.168.1.10", "192.168.1.20", "10.0.0.10"]
    
    security_events = [
        "Failed login attempt from IP: {ip}",
        "Successful login from IP: {ip}",
        "Multiple failed login attempts detected from IP: {ip}",
        "Potential brute force attack from IP: {ip}",
        "Unauthorized access attempt to admin panel",
        "Suspicious file upload detected",
        "SQL injection attempt blocked",
        "Cross-site scripting attempt detected",
        "Port scan detected from IP: {ip}",
        "Malware signature detected in upload"
    ]
    
    with open(filename, "w") as f:
        base_time = datetime.datetime.now() - datetime.timedelta(hours=8)
        
        for i in range(200):
            timestamp = base_time + datetime.timedelta(minutes=i*2)
            
            # 30% chance of suspicious activity
            if random.random() < 0.3:
                ip = random.choice(suspicious_ips)
                event = random.choice([e for e in security_events if "attack" in e or "failed" in e.lower()])
            else:
                ip = random.choice(normal_ips)
                event = random.choice(security_events)
            
            if "{ip}" in event:
                event = event.format(ip=ip)
            
            log_line = f'{timestamp.strftime("%Y-%m-%d %H:%M:%S")} [SECURITY] {event}'
            f.write(log_line + "\n")

def generate_performance_log(filename):
    """Generate performance monitoring log"""
    
    endpoints = ["/api/users", "/api/products", "/api/orders", "/search", "/checkout"]
    
    with open(filename, "w") as f:
        base_time = datetime.datetime.now() - datetime.timedelta(hours=4)
        
        for i in range(400):
            timestamp = base_time + datetime.timedelta(seconds=i*30)
            endpoint = random.choice(endpoints)
            
            # Simulate varying response times
            if endpoint in ["/search", "/checkout"]:
                response_time = random.randint(200, 2000)  # Slower endpoints
            else:
                response_time = random.randint(50, 500)   # Faster endpoints
            
            # Simulate occasional spikes
            if random.random() < 0.05:  # 5% chance of spike
                response_time = random.randint(3000, 8000)
            
            memory_usage = random.randint(40, 85)
            cpu_usage = random.randint(20, 70)
            
            log_line = f'{timestamp.strftime("%Y-%m-%d %H:%M:%S")} [PERF] {endpoint} response_time={response_time}ms memory={memory_usage}% cpu={cpu_usage}%'
            f.write(log_line + "\n")

def create_test_instructions():
    """Create instructions for testing"""
    
    instructions = """# Testing Instructions for Log Analysis BOT

## Sample Log Files Generated

1. **apache_access.log** - Web server access log
   - 1000 entries over 24 hours
   - Mix of successful and error responses
   - Various IP addresses and endpoints

2. **application_errors.log** - Application error log
   - 500 entries over 12 hours
   - Different log levels (INFO, DEBUG, WARN, ERROR, CRITICAL)
   - Various application components

3. **system.log** - System monitoring log
   - 300 entries over 6 hours
   - System services and resource usage
   - Service start/stop events

4. **security.log** - Security events log
   - 200 entries over 8 hours
   - Failed login attempts
   - Potential security threats

5. **performance.log** - Performance monitoring log
   - 400 entries over 4 hours
   - API response times
   - Resource usage metrics

## How to Test

1. Start the Log Analysis BOT:
   ```bash
   python main.py
   ```

2. Open http://localhost:8000 in your browser

3. Upload any of the sample log files

4. Select different analysis options:
   - Try "Error Analysis" with application_errors.log
   - Try "Security Analysis" with security.log
   - Try "Performance Metrics" with performance.log

5. Compare results with different file types

## Expected Results

- **Error Analysis**: Should detect various error patterns
- **Security Analysis**: Should identify suspicious activities
- **Performance Metrics**: Should show response time patterns
- **Trend Analysis**: Should identify usage patterns over time
- **Anomaly Detection**: Should find unusual spikes or patterns

## Troubleshooting

If you encounter issues:
1. Check that your OpenAI API key is configured
2. Verify the backend is running on port 8000
3. Check browser console for JavaScript errors
4. Look at terminal output for Python errors
"""
    
    with open("sample_logs/TESTING_INSTRUCTIONS.md", "w") as f:
        f.write(instructions)
    
    print("✅ Testing instructions created")

if __name__ == "__main__":
    print("🔧 Generating sample log files for testing...")
    generate_sample_logs()
    create_test_instructions()
    print("\n🎉 Sample log files ready for testing!")
    print("📁 Check the 'sample_logs' directory")
    print("📖 Read TESTING_INSTRUCTIONS.md for details")