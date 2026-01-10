# tests/test_main.py
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from competitor_analysis_agent.main import handler


@pytest.mark.asyncio
async def test_handler_returns_response():
    """Test that handler accepts messages and returns a response."""
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    # Mock the run_agent function to return a mock response
    mock_response = MagicMock()
    mock_response.run_id = "test-run-id"
    mock_response.status = "COMPLETED"

    # Mock _initialized to skip initialization and run_agent to return our mock
    with (
        patch("competitor_analysis_agent.main._initialized", True),
        patch("competitor_analysis_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

    # Verify we get a result back
    assert result is not None
    assert result.run_id == "test-run-id"
    assert result.status == "COMPLETED"


@pytest.mark.asyncio
async def test_handler_with_multiple_messages():
    """Test that handler processes multiple messages correctly."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather?"},
    ]

    mock_response = MagicMock()
    mock_response.run_id = "test-run-id-2"

    with (
        patch("competitor_analysis_agent.main._initialized", True),
        patch(
            "competitor_analysis_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response
        ) as mock_run,
    ):
        result = await handler(messages)

    # Verify run_agent was called
    mock_run.assert_called_once_with(messages)
    assert result is not None
    assert result.run_id == "test-run-id-2"


@pytest.mark.asyncio
async def test_handler_initialization():
    """Test that handler initializes on first call."""
    messages = [{"role": "user", "content": "Test"}]

    mock_response = MagicMock()
    mock_response.run_id = "test-run-id-3"

    # Mock the lock to avoid actual locking in tests
    mock_lock = AsyncMock()
    mock_lock.__aenter__ = AsyncMock(return_value=None)
    mock_lock.__aexit__ = AsyncMock(return_value=None)

    # Mock initialize_mcp_tools for the initialization path
    mock_mcp_init = AsyncMock()

    with (
        patch("competitor_analysis_agent.main._initialized", False),
        patch("competitor_analysis_agent.main._init_lock", mock_lock),
        patch("competitor_analysis_agent.main.initialize_mcp_tools", mock_mcp_init),
        patch("competitor_analysis_agent.main.initialize_agent", new_callable=AsyncMock) as mock_init_agent,
        patch("competitor_analysis_agent.main.run_agent", new_callable=AsyncMock, return_value=mock_response),
    ):
        result = await handler(messages)

        # Verify initialization was called
        mock_mcp_init.assert_called_once()
        mock_init_agent.assert_called_once()
        assert result is not None
        assert result.run_id == "test-run-id-3"


@pytest.mark.asyncio
async def test_handler_firecrawl_key_error():
    """Test that handler raises FirecrawlKeyError when API key is missing."""
    messages = [{"role": "user", "content": "Test"}]

    # Mock the lock
    mock_lock = AsyncMock()
    mock_lock.__aenter__ = AsyncMock(return_value=None)
    mock_lock.__aexit__ = AsyncMock(return_value=None)

    with (
        patch("competitor_analysis_agent.main._initialized", False),
        patch("competitor_analysis_agent.main._init_lock", mock_lock),
        patch("competitor_analysis_agent.main.initialize_agent", side_effect=ValueError("Firecrawl API key missing")),
        patch("competitor_analysis_agent.main.run_agent", new_callable=AsyncMock),
        pytest.raises(ValueError, match="Firecrawl API key missing"),
    ):
        await handler(messages)


@pytest.mark.asyncio
async def test_handler_agent_not_ready():
    """Test that handler raises AgentNotReadyError when agent is not initialized."""
    messages = [{"role": "user", "content": "Test"}]

    with (
        patch("competitor_analysis_agent.main._initialized", True),
        patch("competitor_analysis_agent.main.run_agent", side_effect=RuntimeError("Agent not ready")),
        pytest.raises(RuntimeError, match="Agent not ready"),
    ):
        await handler(messages)
