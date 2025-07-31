"""
Trae Agent SDK - Run Function
Python SDK for executing tasks with Trae Agent programmatically.
"""

import asyncio
import os
import traceback
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console

# Import the necessary components from the main trae_agent package
from trae_agent.agent import TraeAgent
from trae_agent.agent.agent_basics import AgentExecution
from trae_agent.utils.cli_console import CLIConsole
from trae_agent.utils.config import Config, load_config

# Load environment variables
_ = load_dotenv()

console = Console()


@dataclass
class TraeAgentSDKResult:
    success: bool
    result: AgentExecution | None
    trajectory_path: str | None
    working_dir: str
    config: Config | None


class TraeAgentSDK:
    """SDK class for Trae Agent operations."""

    def __init__(self, config: Config | None = None):
        """
        Initialize the SDK with optional configuration.

        Args:
            config: Pre-configured Config object. If None, will use default configuration.
        """
        self.config = config
        self.agent = None
        self.cli_console = None

    def create_agent(self, config: Config) -> TraeAgent:
        """
        Create a Trae Agent with the specified configuration.

        Args:
            config: Agent configuration

        Returns:
            TraeAgent object

        Raises:
            Exception: If agent creation fails
        """
        try:
            agent = TraeAgent(config)
            return agent
        except Exception as e:
            console.print(f"[red]Error creating agent: {e}[/red]")
            console.print(traceback.format_exc())
            raise

    def run(
        self,
        task: str,
        patch_path: str | None = None,
        provider: str | None = None,
        model: str | None = None,
        model_base_url: str | None = None,
        api_key: str | None = None,
        max_steps: int | None = None,
        working_dir: str | None = None,
        must_patch: bool = False,
        config_file: str = "trae_config.json",
        trajectory_file: str | None = None,
        verbose: bool = True,
    ) -> TraeAgentSDKResult:
        """
        Run a task using Trae Agent.

        Args:
            task: The task description or path to a file containing the task
            patch_path: Path to patch file
            provider: LLM provider to use
            model: Specific model to use
            api_key: API key for the provider
            max_steps: Maximum number of execution steps
            working_dir: Working directory for the agent
            must_patch: Whether to patch the code
            config_file: Path to configuration file
            trajectory_file: Path to save trajectory file
            verbose: Whether to print verbose output

        Returns:
            Dictionary containing execution results and metadata

        Raises:
            Exception: If task execution fails
        """
        # Set working directory
        if not working_dir:
            working_dir = os.getcwd()

        original_cwd = os.getcwd()
        try:
            os.chdir(working_dir)
            if verbose:
                console.print(f"[blue]Changed working directory to: {working_dir}[/blue]")
        except Exception as e:
            if verbose:
                console.print(f"[red]Error changing directory: {e}[/red]")
            raise
        trajectory_path: str | None = None

        # Load or use existing configuration
        if self.config is None:
            config = load_config(config_file, provider, model, model_base_url, api_key, max_steps)
        else:
            config = self.config
        try:
            # Load task from file if it's a file path
            task_path = Path(task)
            if task_path.exists() and task_path.is_file():
                task = task_path.read_text()

            # Create agent
            agent = self.create_agent(config)

            # Set up trajectory recording
            if trajectory_file:
                trajectory_path = agent.setup_trajectory_recording(trajectory_file)
            else:
                trajectory_path = agent.setup_trajectory_recording()

            # Create CLI Console if verbose output is requested
            if verbose:
                cli_console = CLIConsole(config)
                cli_console.print_task_details(
                    task,
                    working_dir,
                    config.default_provider,
                    config.model_providers[config.default_provider].model,
                    config.max_steps,
                    config_file,
                    trajectory_path,
                )
                agent.set_cli_console(cli_console)

            # Prepare task arguments
            task_args = {
                "project_path": working_dir,
                "issue": task,
                "must_patch": "true" if must_patch else "false",
            }

            if patch_path:
                task_args["patch_path"] = patch_path

            # Execute the task
            agent.new_task(task, task_args)
            result = asyncio.run(agent.execute_task())

            if verbose:
                console.print(f"\n[green]Trajectory saved to: {trajectory_path}[/green]")

            return TraeAgentSDKResult(
                success=True,
                result=result,
                trajectory_path=trajectory_path,
                working_dir=working_dir,
                config=config,
            )

        except KeyboardInterrupt:
            if verbose:
                console.print("\n[yellow]Task execution interrupted by user[/yellow]")
            return TraeAgentSDKResult(
                success=False,
                result=None,
                trajectory_path=trajectory_path if "trajectory_path" in locals() else None,
                working_dir=working_dir,
                config=config,
            )
        except Exception as e:
            if verbose:
                console.print(f"\n[red]Unexpected error: {e}[/red]")
                console.print(traceback.format_exc())
            return TraeAgentSDKResult(
                success=False,
                result=None,
                trajectory_path=trajectory_path if "trajectory_path" in locals() else None,
                working_dir=working_dir,
                config=config,
            )
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
