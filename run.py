#!/usr/bin/env python
"""
Simple entry point to run the OpenAI-compatible API server
"""
import logging
import socket
import uvicorn
from app.config.settings import API_PORT, API_HOST

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Check if IPv6 is supported
    has_ipv6 = False
    try:
        socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        has_ipv6 = True
        logger.info("IPv6 is supported")
    except OSError:
        logger.info("IPv6 is not supported")
    
    # Start server with IPv6 dual stack enabled if supported
    host = "::" if has_ipv6 else API_HOST
    
    logger.info(f"Starting server on {host}:{API_PORT}")
    uvicorn.run(
        "main:app", 
        host=host,
        port=API_PORT, 
        reload=True
    ) 