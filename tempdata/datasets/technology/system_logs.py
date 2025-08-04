"""
System logs generator

Generates realistic system log data with log level distributions, error patterns,
service correlations, and realistic timestamp patterns.
"""

import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from ...core.base_generator import BaseGenerator


class SystemLogsGenerator(BaseGenerator):
    """
    Generator for realistic system log data
    
    Creates system log datasets with log level distributions, error patterns,
    service correlations, and realistic timestamp patterns.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_log_level_distributions()
        self._setup_service_configurations()
        self._setup_error_patterns()
        self._setup_message_templates()
        self._setup_correlation_patterns()
    
    def _setup_log_level_distributions(self):
        """Setup realistic log level distributions"""
        self.log_levels = {
            'DEBUG': {
                'probability': 0.15,
                'severity': 1,
                'typical_services': ['api', 'web_server', 'application'],
                'time_clustering': False  # Debug logs are usually evenly distributed
            },
            'INFO': {
                'probability': 0.55,
                'severity': 2,
                'typical_services': ['api', 'web_server', 'application', 'database', 'cache'],
                'time_clustering': False
            },
            'WARN': {
                'probability': 0.20,
                'severity': 3,
                'typical_services': ['api', 'database', 'cache', 'queue'],
                'time_clustering': True  # Warnings often cluster during issues
            },
            'ERROR': {
                'probability': 0.08,
                'severity': 4,
                'typical_services': ['api', 'database', 'payment', 'external_service'],
                'time_clustering': True  # Errors cluster during incidents
            },
            'FATAL': {
                'probability': 0.02,
                'severity': 5,
                'typical_services': ['database', 'payment', 'core_service'],
                'time_clustering': True  # Fatal errors are rare but cluster during major incidents
            }
        }
    
    def _setup_service_configurations(self):
        """Setup service configurations and their characteristics"""
        self.services = {
            'api': {
                'probability': 0.30,
                'log_frequency': 'high',  # logs per minute
                'error_rate': 0.05,
                'typical_operations': ['GET', 'POST', 'PUT', 'DELETE'],
                'response_time_range': (10, 2000),  # milliseconds
                'common_endpoints': ['/users', '/orders', '/products', '/auth', '/health']
            },
            'web_server': {
                'probability': 0.25,
                'log_frequency': 'high',
                'error_rate': 0.03,
                'typical_operations': ['request', 'response', 'static_file'],
                'response_time_range': (5, 500),
                'common_endpoints': ['/index.html', '/assets', '/api', '/login', '/dashboard']
            },
            'database': {
                'probability': 0.20,
                'log_frequency': 'medium',
                'error_rate': 0.02,
                'typical_operations': ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CONNECT'],
                'response_time_range': (1, 5000),
                'common_endpoints': ['users', 'orders', 'products', 'sessions', 'logs']
            },
            'cache': {
                'probability': 0.10,
                'log_frequency': 'medium',
                'error_rate': 0.005,  # Reduced error rate for cache
                'typical_operations': ['GET', 'SET', 'DELETE', 'EXPIRE'],
                'response_time_range': (1, 50),
                'common_endpoints': ['user_session', 'product_cache', 'api_cache']
            },
            'queue': {
                'probability': 0.08,
                'log_frequency': 'low',
                'error_rate': 0.04,
                'typical_operations': ['ENQUEUE', 'DEQUEUE', 'PROCESS', 'RETRY'],
                'response_time_range': (10, 30000),
                'common_endpoints': ['email_queue', 'notification_queue', 'processing_queue']
            },
            'payment': {
                'probability': 0.04,
                'log_frequency': 'low',
                'error_rate': 0.08,
                'typical_operations': ['CHARGE', 'REFUND', 'VALIDATE', 'WEBHOOK'],
                'response_time_range': (100, 10000),
                'common_endpoints': ['stripe', 'paypal', 'bank_transfer']
            },
            'external_service': {
                'probability': 0.03,
                'log_frequency': 'low',
                'error_rate': 0.12,
                'typical_operations': ['API_CALL', 'WEBHOOK', 'SYNC'],
                'response_time_range': (200, 15000),
                'common_endpoints': ['third_party_api', 'webhook_endpoint', 'data_sync']
            }
        }
    
    def _setup_error_patterns(self):
        """Setup error patterns and their characteristics"""
        self.error_patterns = {
            'connection_timeout': {
                'probability': 0.25,
                'services': ['database', 'external_service', 'cache'],
                'clustering_factor': 0.7,  # How likely to occur in clusters
                'duration_minutes': (5, 30),  # How long the issue typically lasts
                'message_templates': [
                    'Connection timeout to {service} after {timeout}ms',
                    'Failed to connect to {service}: timeout',
                    'Connection pool exhausted for {service}'
                ]
            },
            'authentication_failure': {
                'probability': 0.20,
                'services': ['api', 'web_server'],
                'clustering_factor': 0.3,
                'duration_minutes': (1, 5),
                'message_templates': [
                    'Authentication failed for user {user_id}',
                    'Invalid token provided',
                    'Session expired for user {user_id}'
                ]
            },
            'resource_exhaustion': {
                'probability': 0.15,
                'services': ['database', 'api', 'cache'],
                'clustering_factor': 0.8,
                'duration_minutes': (10, 60),
                'message_templates': [
                    'Memory usage exceeded threshold: {percentage}%',
                    'CPU usage critical: {percentage}%',
                    'Disk space low: {percentage}% remaining'
                ]
            },
            'validation_error': {
                'probability': 0.15,
                'services': ['api', 'payment'],
                'clustering_factor': 0.2,
                'duration_minutes': (1, 3),
                'message_templates': [
                    'Invalid input data: {field} is required',
                    'Validation failed for {field}: {reason}',
                    'Data format error in {field}'
                ]
            },
            'service_unavailable': {
                'probability': 0.10,
                'services': ['external_service', 'payment', 'queue'],
                'clustering_factor': 0.9,
                'duration_minutes': (15, 120),
                'message_templates': [
                    'Service {service} is unavailable',
                    'HTTP 503: Service temporarily unavailable',
                    'Upstream service {service} not responding'
                ]
            },
            'data_corruption': {
                'probability': 0.05,
                'services': ['database', 'queue'],
                'clustering_factor': 0.6,
                'duration_minutes': (30, 180),
                'message_templates': [
                    'Data integrity check failed for table {table}',
                    'Corrupted data detected in {location}',
                    'Checksum mismatch in {file}'
                ]
            },
            'rate_limit_exceeded': {
                'probability': 0.10,
                'services': ['api', 'external_service'],
                'clustering_factor': 0.5,
                'duration_minutes': (5, 15),
                'message_templates': [
                    'Rate limit exceeded for endpoint {endpoint}',
                    'Too many requests from IP {ip}',
                    'API quota exceeded for user {user_id}'
                ]
            }
        }
    
    def _setup_message_templates(self):
        """Setup message templates for different log levels and operations"""
        self.message_templates = {
            'DEBUG': {
                'api': [
                    'Processing request to {endpoint}',
                    'Validating input parameters for {operation}',
                    'Database query executed: {query}',
                    'Cache lookup for key: {key}',
                    'Response prepared for request {request_id}'
                ],
                'web_server': [
                    'Serving static file: {file}',
                    'Processing HTTP {method} request',
                    'Session created for user {user_id}',
                    'Middleware {middleware} executed',
                    'Route matched: {route}'
                ],
                'database': [
                    'Connection established to database {db_name}',
                    'Query plan generated for {query}',
                    'Index scan on table {table}',
                    'Transaction started: {transaction_id}',
                    'Connection returned to pool'
                ]
            },
            'INFO': {
                'api': [
                    'Request completed: {method} {endpoint} - {status_code} in {response_time}ms',
                    'User {user_id} authenticated successfully',
                    'Data retrieved for request {request_id}',
                    'Cache hit for key: {key}',
                    'Background job {job_id} started'
                ],
                'web_server': [
                    'Server started on port {port}',
                    'Request processed: {method} {path} - {status_code}',
                    'User session established: {session_id}',
                    'Health check passed',
                    'Configuration reloaded'
                ],
                'database': [
                    'Database connection pool initialized',
                    'Query executed successfully: {query_type} on {table}',
                    'Backup completed for database {db_name}',
                    'Index rebuilt on table {table}',
                    'Statistics updated for table {table}'
                ]
            },
            'WARN': {
                'api': [
                    'Slow response detected: {endpoint} took {response_time}ms',
                    'High memory usage: {percentage}%',
                    'Deprecated endpoint accessed: {endpoint}',
                    'Rate limiting applied to user {user_id}',
                    'Cache miss rate high: {percentage}%'
                ],
                'database': [
                    'Long running query detected: {duration}ms',
                    'Connection pool near capacity: {used}/{total}',
                    'Table {table} requires optimization',
                    'Deadlock detected and resolved',
                    'Disk space warning: {percentage}% used'
                ],
                'cache': [
                    'Cache eviction rate high: {rate} per second',
                    'Memory usage warning: {percentage}%',
                    'Connection timeout to cache server',
                    'Cache key expired: {key}',
                    'Replication lag detected: {lag}ms'
                ]
            },
            'ERROR': [
                'Failed to process request: {error_message}',
                'Database connection failed: {error_code}',
                'External service call failed: {service} - {error}',
                'Payment processing failed: {transaction_id}',
                'File operation failed: {operation} on {file}',
                'Queue processing error: {queue_name} - {error}',
                'Authentication service unavailable',
                'Data validation failed: {details}'
            ],
            'FATAL': [
                'System out of memory - shutting down',
                'Database corruption detected - service stopped',
                'Critical security breach detected',
                'Payment gateway connection lost',
                'Core service failure - initiating failover',
                'Disk full - cannot continue operations',
                'Network partition detected - cluster unstable'
            ]
        }
    
    def _setup_correlation_patterns(self):
        """Setup patterns for correlated log events"""
        self.correlation_patterns = {
            'incident_cascade': {
                'trigger_services': ['database', 'payment'],
                'affected_services': ['api', 'web_server', 'queue'],
                'cascade_delay_seconds': (10, 60),
                'duration_multiplier': 2.0
            },
            'deployment_issues': {
                'trigger_services': ['api', 'web_server'],
                'affected_services': ['database', 'cache'],
                'cascade_delay_seconds': (5, 30),
                'duration_multiplier': 1.5
            },
            'traffic_spike': {
                'trigger_services': ['web_server', 'api'],
                'affected_services': ['database', 'cache', 'queue'],
                'cascade_delay_seconds': (1, 10),
                'duration_multiplier': 1.2
            }
        }
    
    def generate(self, rows: int, **kwargs) -> pd.DataFrame:
        """
        Generate system logs dataset
        
        Args:
            rows: Number of log entries to generate
            **kwargs: Additional parameters (time_series, date_range, etc.)
            
        Returns:
            pd.DataFrame: Generated system logs data with realistic patterns
        """
        time_series = kwargs.get('time_series', False)
        date_range = kwargs.get('date_range', None)
        
        data = []
        incident_tracker = {}  # Track ongoing incidents for clustering
        
        # Generate base timestamp range
        if time_series and date_range:
            start_date, end_date = date_range
            base_time = self.faker.date_time_between(start_date=start_date, end_date=end_date)
        else:
            base_time = self.faker.date_time_this_month()
        
        for i in range(rows):
            log_entry = self._generate_log_entry(
                i, base_time, incident_tracker, time_series
            )
            data.append(log_entry)
        
        df = pd.DataFrame(data)
        return self._apply_realistic_patterns(df)
    
    def _generate_log_entry(self, index: int, base_time: datetime, 
                           incident_tracker: Dict, time_series: bool) -> Dict[str, Any]:
        """Generate individual log entry"""
        
        # Determine if this should be part of an incident cluster
        is_incident_log = self._should_create_incident_log(incident_tracker)
        
        if is_incident_log:
            log_level, service, error_pattern = self._select_incident_characteristics(incident_tracker)
        else:
            # Normal log generation - use original distribution but bias service selection for errors
            log_level = self._select_weighted_choice(
                list(self.log_levels.keys()),
                [self.log_levels[level]['probability'] for level in self.log_levels.keys()]
            )
            
            # Select service based on log level - error-prone services more likely for error logs
            if log_level in ['ERROR', 'FATAL']:
                # Strong bias towards services with higher error rates for error logs
                service_weights = [
                    self.services[svc]['probability'] * (1 + self.services[svc]['error_rate'] * 50)
                    for svc in self.services.keys()
                ]
                service = self._select_weighted_choice(
                    list(self.services.keys()),
                    service_weights
                )
                error_pattern = self._select_error_pattern()
            elif log_level == 'WARN':
                # Moderate bias for warnings
                service_weights = [
                    self.services[svc]['probability'] * (1 + self.services[svc]['error_rate'] * 20)
                    for svc in self.services.keys()
                ]
                service = self._select_weighted_choice(
                    list(self.services.keys()),
                    service_weights
                )
                error_pattern = None
            else:
                # Normal service selection for INFO/DEBUG
                service = self._select_weighted_choice(
                    list(self.services.keys()),
                    [self.services[svc]['probability'] for svc in self.services.keys()]
                )
                error_pattern = None
        
        # Generate timestamp with realistic clustering
        timestamp = self._generate_timestamp(
            base_time, index, log_level, service, time_series, is_incident_log
        )
        
        # Generate log message
        message = self._generate_log_message(log_level, service, error_pattern)
        
        # Generate additional fields
        log_entry = {
            'timestamp': timestamp,
            'log_level': log_level,
            'service': service,
            'message': message,
            'thread_id': f'thread-{self.faker.random_int(1, 50):02d}',
            'process_id': self.faker.random_int(1000, 9999),
            'host': self._generate_hostname(service),
            'source_file': self._generate_source_file(service),
            'line_number': self.faker.random_int(1, 1000),
            'request_id': f'req-{self.faker.uuid4()[:8]}' if service in ['api', 'web_server'] else None,
            'user_id': f'user-{self.faker.random_int(1, 10000):06d}' if self.faker.random.random() < 0.3 else None,
            'session_id': f'sess-{self.faker.uuid4()[:12]}' if service in ['api', 'web_server'] and self.faker.random.random() < 0.4 else None,
            'response_time_ms': self._generate_response_time(service, log_level),
            'status_code': self._generate_status_code(service, log_level),
            'ip_address': self.faker.ipv4() if service in ['api', 'web_server'] else None,
            'user_agent': self.faker.user_agent() if service == 'web_server' else None,
            'error_code': self._generate_error_code(log_level, error_pattern),
            'stack_trace': self._generate_stack_trace(log_level) if log_level in ['ERROR', 'FATAL'] and self.faker.random.random() < 0.3 else None
        }
        
        # Update incident tracker
        if error_pattern and log_level in ['ERROR', 'FATAL']:
            self._update_incident_tracker(incident_tracker, error_pattern, timestamp, service)
        
        return log_entry
    
    def _should_create_incident_log(self, incident_tracker: Dict) -> bool:
        """Determine if this log should be part of an incident cluster"""
        # Check if any incidents are currently active
        current_time = datetime.now()
        active_incidents = []
        
        for incident_id, incident_data in incident_tracker.items():
            if current_time < incident_data['end_time']:
                active_incidents.append(incident_data)
        
        if not active_incidents:
            # No active incidents, small chance to start a new one
            return self.faker.random.random() < 0.05
        
        # If there are active incidents, higher chance to add to them
        return self.faker.random.random() < 0.3
    
    def _select_incident_characteristics(self, incident_tracker: Dict) -> Tuple[str, str, str]:
        """Select characteristics for an incident log"""
        # Prefer ERROR and WARN levels during incidents
        log_level = self._select_weighted_choice(
            ['WARN', 'ERROR', 'FATAL'],
            [0.6, 0.35, 0.05]
        )
        
        # Select error pattern
        error_pattern = self._select_error_pattern()
        
        # Select service based on error pattern
        pattern_info = self.error_patterns[error_pattern]
        service = self.faker.random_element(pattern_info['services'])
        
        return log_level, service, error_pattern
    
    def _select_error_pattern(self) -> str:
        """Select error pattern based on probabilities"""
        return self._select_weighted_choice(
            list(self.error_patterns.keys()),
            [self.error_patterns[pattern]['probability'] for pattern in self.error_patterns.keys()]
        )
    
    def _generate_timestamp(self, base_time: datetime, index: int, log_level: str, 
                           service: str, time_series: bool, is_incident_log: bool) -> datetime:
        """Generate realistic timestamp with clustering patterns"""
        if time_series:
            # For time series, spread logs across the time range
            time_offset = timedelta(seconds=index * self.faker.random_int(1, 10))
        else:
            # For non-time series, use more random distribution
            time_offset = timedelta(
                seconds=self.faker.random_int(0, 86400),  # Within 24 hours
                microseconds=self.faker.random_int(0, 999999)
            )
        
        timestamp = base_time + time_offset
        
        # Add clustering for error logs and incidents
        if is_incident_log or log_level in ['ERROR', 'FATAL']:
            # Cluster around certain times (business hours have more activity)
            hour = timestamp.hour
            if 9 <= hour <= 17:  # Business hours
                # Add small random offset for clustering
                cluster_offset = timedelta(seconds=self.faker.random_int(-300, 300))
                timestamp += cluster_offset
        
        return timestamp
    
    def _generate_log_message(self, log_level: str, service: str, error_pattern: str = None) -> str:
        """Generate realistic log message"""
        if error_pattern:
            # Use error pattern templates
            pattern_info = self.error_patterns[error_pattern]
            template = self.faker.random_element(pattern_info['message_templates'])
            
            # Fill in template variables
            message = template.format(
                service=service,
                timeout=self.faker.random_int(1000, 30000),
                user_id=f'user-{self.faker.random_int(1, 10000):06d}',
                percentage=self.faker.random_int(80, 99),
                field=self.faker.random_element(['email', 'password', 'amount', 'date']),
                reason=self.faker.random_element(['invalid format', 'too long', 'required']),
                endpoint=self.faker.random_element(self.services[service]['common_endpoints']),
                ip=self.faker.ipv4(),
                table=self.faker.random_element(['users', 'orders', 'products']),
                location=f'file_{self.faker.random_int(1, 100)}.dat',
                file=f'data_{self.faker.random_int(1, 1000)}.log'
            )
        else:
            # Use normal message templates
            if log_level in ['ERROR', 'FATAL']:
                template = self.faker.random_element(self.message_templates[log_level])
                message = template.format(
                    error_message=self.faker.sentence(),
                    error_code=f'ERR_{self.faker.random_int(1000, 9999)}',
                    service=service,
                    error=self.faker.word(),
                    transaction_id=f'txn_{self.faker.uuid4()[:8]}',
                    operation=self.faker.random_element(['read', 'write', 'delete']),
                    file=f'{self.faker.word()}.{self.faker.random_element(["log", "dat", "tmp"])}',
                    queue_name=f'{service}_queue',
                    details=self.faker.sentence()
                )
            else:
                # Use service-specific templates
                service_templates = self.message_templates.get(log_level, {}).get(service, 
                    self.message_templates[log_level].get('api', ['Generic log message']))
                
                template = self.faker.random_element(service_templates)
                
                # Fill in common template variables
                message = template.format(
                    endpoint=self.faker.random_element(self.services[service]['common_endpoints']),
                    operation=self.faker.random_element(self.services[service]['typical_operations']),
                    method=self.faker.random_element(['GET', 'POST', 'PUT', 'DELETE']),
                    status_code=self._generate_status_code(service, log_level),
                    response_time=self._generate_response_time(service, log_level),
                    user_id=f'user-{self.faker.random_int(1, 10000):06d}',
                    request_id=f'req-{self.faker.uuid4()[:8]}',
                    key=f'{service}_{self.faker.word()}',
                    port=self.faker.random_int(8000, 9000),
                    path=f'/{self.faker.uri_path()}',
                    session_id=f'sess-{self.faker.uuid4()[:12]}',
                    db_name=f'{service}_db',
                    query=f'SELECT * FROM {self.faker.word()}',
                    table=self.faker.random_element(['users', 'orders', 'products', 'sessions']),
                    transaction_id=f'txn_{self.faker.uuid4()[:8]}',
                    file=f'{self.faker.word()}.{self.faker.random_element(["html", "css", "js", "png"])}',
                    middleware=f'{self.faker.word()}_middleware',
                    route=f'/{self.faker.uri_path()}',
                    job_id=f'job_{self.faker.uuid4()[:8]}',
                    percentage=self.faker.random_int(10, 95),
                    used=self.faker.random_int(50, 95),
                    total=100,
                    duration=self.faker.random_int(1000, 10000),
                    rate=self.faker.random_int(10, 100),
                    lag=self.faker.random_int(10, 1000),
                    query_type=self.faker.random_element(['SELECT', 'INSERT', 'UPDATE', 'DELETE'])
                )
        
        return message
    
    def _generate_hostname(self, service: str) -> str:
        """Generate realistic hostname based on service"""
        env = self.faker.random_element(['prod', 'staging', 'dev'])
        instance = self.faker.random_int(1, 10)
        return f'{service}-{env}-{instance:02d}.company.com'
    
    def _generate_source_file(self, service: str) -> str:
        """Generate realistic source file name"""
        file_patterns = {
            'api': ['controller', 'service', 'middleware', 'handler'],
            'web_server': ['server', 'router', 'static', 'proxy'],
            'database': ['connection', 'query', 'migration', 'backup'],
            'cache': ['redis', 'memcache', 'storage', 'eviction'],
            'queue': ['worker', 'processor', 'scheduler', 'consumer'],
            'payment': ['gateway', 'processor', 'webhook', 'validator'],
            'external_service': ['client', 'adapter', 'sync', 'webhook']
        }
        
        pattern = self.faker.random_element(file_patterns.get(service, ['main']))
        return f'{pattern}.py'
    
    def _generate_response_time(self, service: str, log_level: str) -> int:
        """Generate realistic response time based on service and log level"""
        service_info = self.services[service]
        min_time, max_time = service_info['response_time_range']
        
        # Adjust based on log level
        if log_level in ['ERROR', 'FATAL']:
            # Errors often have longer response times
            multiplier = self.faker.random.uniform(2.0, 3.0)  # Reduced multiplier
        elif log_level == 'WARN':
            # Warnings might indicate slower responses
            multiplier = self.faker.random.uniform(1.5, 2.0)  # Reduced multiplier
        else:
            multiplier = self.faker.random.uniform(0.8, 1.2)
        
        response_time = int((min_time + self.faker.random_int(0, max_time - min_time)) * multiplier)
        # Cap at 60 seconds (60000ms) to ensure reasonable response times
        return max(1, min(response_time, 60000))
    
    def _generate_status_code(self, service: str, log_level: str) -> int:
        """Generate realistic HTTP status code"""
        if service not in ['api', 'web_server']:
            return None
        
        if log_level == 'ERROR':
            return self.faker.random_element([400, 401, 403, 404, 409, 422, 500, 502, 503])
        elif log_level == 'FATAL':
            return self.faker.random_element([500, 502, 503, 504])
        elif log_level == 'WARN':
            return self.faker.random_element([200, 201, 202, 400, 401, 404])
        else:
            return self.faker.random_element([200, 201, 202, 204, 301, 302, 304])
    
    def _generate_error_code(self, log_level: str, error_pattern: str = None) -> str:
        """Generate error code for error logs"""
        if log_level not in ['ERROR', 'FATAL']:
            return None
        
        if error_pattern:
            error_codes = {
                'connection_timeout': ['CONN_TIMEOUT', 'NET_UNREACHABLE', 'CONN_REFUSED'],
                'authentication_failure': ['AUTH_FAILED', 'TOKEN_INVALID', 'SESSION_EXPIRED'],
                'resource_exhaustion': ['MEM_EXHAUSTED', 'CPU_OVERLOAD', 'DISK_FULL'],
                'validation_error': ['INVALID_INPUT', 'VALIDATION_FAILED', 'FORMAT_ERROR'],
                'service_unavailable': ['SVC_UNAVAILABLE', 'UPSTREAM_DOWN', 'CIRCUIT_OPEN'],
                'data_corruption': ['DATA_CORRUPT', 'CHECKSUM_FAIL', 'INTEGRITY_ERROR'],
                'rate_limit_exceeded': ['RATE_LIMIT', 'QUOTA_EXCEEDED', 'TOO_MANY_REQUESTS']
            }
            
            codes = error_codes.get(error_pattern, ['GENERIC_ERROR'])
            return self.faker.random_element(codes)
        
        # Generic error codes
        return f'ERR_{self.faker.random_int(1000, 9999)}'
    
    def _generate_stack_trace(self, log_level: str) -> str:
        """Generate realistic stack trace for error logs"""
        if log_level not in ['ERROR', 'FATAL']:
            return None
        
        stack_lines = []
        depth = self.faker.random_int(3, 8)
        
        for i in range(depth):
            file_name = f'{self.faker.word()}.py'
            line_num = self.faker.random_int(1, 500)
            function_name = f'{self.faker.word()}_{self.faker.word()}'
            stack_lines.append(f'  File "{file_name}", line {line_num}, in {function_name}')
            
            if i == 0:  # Add the actual error line
                error_line = f'    {self.faker.sentence()}'
                stack_lines.append(error_line)
        
        return '\n'.join(stack_lines)
    
    def _update_incident_tracker(self, incident_tracker: Dict, error_pattern: str, 
                                timestamp: datetime, service: str):
        """Update incident tracker for clustering"""
        pattern_info = self.error_patterns[error_pattern]
        
        # Create incident ID
        incident_id = f'{error_pattern}_{service}_{timestamp.strftime("%Y%m%d_%H%M")}'
        
        if incident_id not in incident_tracker:
            # New incident
            duration_min = self.faker.random_int(*pattern_info['duration_minutes'])
            incident_tracker[incident_id] = {
                'start_time': timestamp,
                'end_time': timestamp + timedelta(minutes=duration_min),
                'error_pattern': error_pattern,
                'primary_service': service,
                'log_count': 1
            }
        else:
            # Existing incident
            incident_tracker[incident_id]['log_count'] += 1
    
    def _select_weighted_choice(self, choices: List, weights: List) -> Any:
        """Select item from choices based on weights using seeded random"""
        total_weight = sum(weights)
        rand_val = self.faker.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for choice, weight in zip(choices, weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return choice
        
        return choices[-1]  # Fallback
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Sort by timestamp for chronological order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived metrics
        data['severity_score'] = data['log_level'].map({
            'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'FATAL': 5
        })
    
    def _generate_stack_trace(self, log_level: str) -> str:
        """Generate realistic stack trace for error logs"""
        if log_level not in ['ERROR', 'FATAL']:
            return None
        
        stack_lines = []
        depth = self.faker.random_int(3, 8)
        
        for i in range(depth):
            file_name = f'{self.faker.word()}.py'
            line_num = self.faker.random_int(1, 500)
            function_name = f'{self.faker.word()}_{self.faker.word()}'
            stack_lines.append(f'  File "{file_name}", line {line_num}, in {function_name}')
            
            if i == 0:  # Add the actual error line
                error_line = f'    {self.faker.sentence()}'
                stack_lines.append(error_line)
        
        return '\n'.join(stack_lines)
    
    def _update_incident_tracker(self, incident_tracker: Dict, error_pattern: str, 
                                timestamp: datetime, service: str):
        """Update incident tracker for clustering"""
        pattern_info = self.error_patterns[error_pattern]
        
        # Create incident ID
        incident_id = f'{error_pattern}_{service}_{timestamp.strftime("%Y%m%d_%H%M")}'
        
        if incident_id not in incident_tracker:
            # New incident
            duration_min = self.faker.random_int(*pattern_info['duration_minutes'])
            incident_tracker[incident_id] = {
                'start_time': timestamp,
                'end_time': timestamp + timedelta(minutes=duration_min),
                'error_pattern': error_pattern,
                'primary_service': service,
                'log_count': 1
            }
        else:
            # Existing incident
            incident_tracker[incident_id]['log_count'] += 1
    
    def _select_weighted_choice(self, choices: List, weights: List) -> Any:
        """Select item from choices based on weights using seeded random"""
        total_weight = sum(weights)
        rand_val = self.faker.random.uniform(0, total_weight)
        
        cumulative_weight = 0
        for choice, weight in zip(choices, weights):
            cumulative_weight += weight
            if rand_val <= cumulative_weight:
                return choice
        
        return choices[-1]  # Fallback
    
    def _apply_realistic_patterns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply additional realistic patterns to the generated data"""
        # Sort by timestamp for chronological order
        data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Add derived metrics
        data['severity_score'] = data['log_level'].map({
            'DEBUG': 1, 'INFO': 2, 'WARN': 3, 'ERROR': 4, 'FATAL': 5
        })
        
        # Add time-based features
        data['hour_of_day'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        data['is_business_hours'] = ((data['hour_of_day'] >= 9) & (data['hour_of_day'] <= 17)).astype(bool)
        data['is_weekend'] = (data['day_of_week'] >= 5).astype(bool)
        
        # Add log frequency metrics (logs per minute)
        data['timestamp_minute'] = data['timestamp'].dt.floor('min')
        minute_counts = data.groupby('timestamp_minute').size()
        data = data.merge(
            minute_counts.rename('logs_per_minute'), 
            left_on='timestamp_minute', 
            right_index=True
        )
        
        # Add service health indicators
        service_error_rates = data.groupby('service')['log_level'].apply(
            lambda x: (x.isin(['ERROR', 'FATAL'])).mean()
        ).rename('service_error_rate')
        
        data = data.merge(service_error_rates, left_on='service', right_index=True)
        
        # Clean up temporary columns
        data = data.drop('timestamp_minute', axis=1)
        
        return data
     