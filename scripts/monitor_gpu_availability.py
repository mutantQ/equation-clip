#!/usr/bin/env python3
"""
GPU Availability Monitor for Hyperbolic

Continuously monitors Hyperbolic for available GPU instances
and notifies when GPUs become available.
"""

import time
import json
import logging
from datetime import datetime
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GPUMonitor:
    """Monitor Hyperbolic GPU availability."""

    def __init__(self, check_interval: int = 300, notify_callback=None):
        """
        Initialize GPU monitor.

        Args:
            check_interval: Seconds between checks (default: 5 minutes)
            notify_callback: Function to call when GPU becomes available
        """
        self.check_interval = check_interval
        self.notify_callback = notify_callback
        self.last_check_time = None
        self.availability_log = []

    def check_availability(self) -> dict:
        """
        Check GPU availability using Hyperbolic MCP.

        Returns:
            Dictionary with availability status
        """
        try:
            # This will be called through Claude Code's MCP integration
            # For now, return a placeholder that Claude Code will handle
            logger.info("Checking GPU availability on Hyperbolic...")

            # Claude Code will need to call mcp__hyperbolic-gpu__list-available-gpus
            # and update this function with the actual implementation

            result = {
                'timestamp': datetime.now().isoformat(),
                'available': False,
                'gpus': [],
                'message': 'Waiting for GPU availability check via MCP'
            }

            self.last_check_time = datetime.now()
            self.availability_log.append(result)

            return result

        except Exception as e:
            logger.error(f"Error checking GPU availability: {str(e)}")
            return {
                'timestamp': datetime.now().isoformat(),
                'available': False,
                'error': str(e)
            }

    def notify(self, message: str, gpu_info: dict = None):
        """
        Send notification about GPU availability.

        Args:
            message: Notification message
            gpu_info: GPU information dictionary
        """
        logger.info("=" * 60)
        logger.info("🎯 GPU AVAILABILITY ALERT")
        logger.info("=" * 60)
        logger.info(message)

        if gpu_info:
            logger.info(f"\nAvailable GPUs: {len(gpu_info)}")
            for gpu in gpu_info[:3]:  # Show first 3
                logger.info(f"  - {gpu.get('name', 'Unknown')} | "
                          f"Cluster: {gpu.get('cluster_name', 'N/A')}")

        logger.info("=" * 60)

        # Call custom callback if provided
        if self.notify_callback:
            self.notify_callback(message, gpu_info)

    def save_log(self, log_file: Path = None):
        """Save availability log to file."""
        if log_file is None:
            log_file = Path('./logs/gpu_availability.json')

        log_file.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, 'w') as f:
            json.dump(self.availability_log, f, indent=2)

        logger.info(f"Availability log saved to {log_file}")

    def run_continuous(self, max_checks: int = None):
        """
        Run continuous monitoring.

        Args:
            max_checks: Maximum number of checks (None = infinite)
        """
        logger.info("Starting GPU availability monitoring...")
        logger.info(f"Check interval: {self.check_interval} seconds")

        check_count = 0

        try:
            while max_checks is None or check_count < max_checks:
                # NOTE: This function needs to be enhanced by Claude Code
                # to actually call the MCP tool and get real availability data

                logger.info(f"Check #{check_count + 1} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info("⏳ Waiting for Claude Code to check GPU availability via MCP...")

                # Placeholder for actual check
                # Claude Code should call mcp__hyperbolic-gpu__list-available-gpus here
                result = self.check_availability()

                check_count += 1

                # Sleep until next check
                if max_checks is None or check_count < max_checks:
                    logger.info(f"Next check in {self.check_interval} seconds...\n")
                    time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("\nMonitoring stopped by user")
        finally:
            self.save_log()
            logger.info(f"Total checks performed: {check_count}")


def main():
    """Main execution function."""
    # Configuration
    CHECK_INTERVAL = 300  # 5 minutes

    def custom_notify(message, gpu_info):
        """Custom notification handler."""
        # Could add email, Slack, or other notifications here
        print(f"\n🔔 NOTIFICATION: {message}\n")

    monitor = GPUMonitor(
        check_interval=CHECK_INTERVAL,
        notify_callback=custom_notify
    )

    # Run continuous monitoring
    logger.info("""
    ╔═══════════════════════════════════════════════════════╗
    ║     Hyperbolic GPU Availability Monitor               ║
    ║                                                       ║
    ║  This script monitors GPU availability continuously  ║
    ║  Press Ctrl+C to stop                                ║
    ╚═══════════════════════════════════════════════════════╝
    """)

    monitor.run_continuous()


if __name__ == "__main__":
    main()
