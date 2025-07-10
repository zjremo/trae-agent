"""
Trae Agent SDK - Run Function
Python SDK for executing tasks with Trae Agent programmatically.
"""

import asyncio
import os
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from rich.console import Console

# Import the necessary components from the main trae_agent package
from trae_agent.agent import TraeAgent
from trae_agent.utils.cli_console import CLIConsole
from trae_agent.utils.config import Config, resolve_config_value

# Load environment variables
_ = load_dotenv()

console = Console()


class TraeAgentSDK:
    """SDK class for Trae Agent operations."""

    def __init__(self, config: Optional[Config] = None):
        """
        Initialize the SDK with optional configuration.

        Args:
            config: Pre-configured Config object. If None, will use default configuration.
        """
        self.config = config
        self.agent = None
        self.cli_console = None

    def load_config(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        config_file: str = "trae_config.json",
        max_steps: Optional[int] = 20,
    ) -> Config:
        """
        Load configuration for the agent.

        Args:
            provider: LLM provider to use (default: openai)
            model: Specific model to use
            api_key: API key for the provider
            config_file: Path to configuration file
            max_steps: Maximum number of execution steps

        Returns:
            Config object with resolved configuration
        """
        config = Config(config_file)

        # Resolve model provider
        resolved_provider = resolve_config_value(provider, config.default_provider) or "openai"
        config.default_provider = str(resolved_provider)

        # Resolve configuration values with overrides
        resolved_model = resolve_config_value(
            model, config.model_providers[str(resolved_provider)].model
        )

        model_parameters = config.model_providers[str(resolved_provider)]
        if resolved_model is not None:
            model_parameters.model = str(resolved_model)

        # Map providers to their environment variable names
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "azure": "AZURE_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
            "doubao": "DOUBAO_API_KEY",
            "google": "GOOGLE_API_KEY",
        }

        resolved_api_key = resolve_config_value(
            api_key,
            config.model_providers[str(resolved_provider)].api_key,
            env_var_map.get(str(resolved_provider)),
        )

        if resolved_api_key is not None:
            model_parameters.api_key = str(resolved_api_key)

        resolved_max_steps = resolve_config_value(max_steps, config.max_steps)
        if resolved_max_steps is not None:
            config.max_steps = int(resolved_max_steps)

        return config

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
        patch_path: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        max_steps: Optional[int] = None,
        working_dir: Optional[str] = None,
        must_patch: bool = False,
        config_file: str = "trae_config.json",
        trajectory_file: Optional[str] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
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

        try:
            # Load task from file if it's a file path
            task_path = Path(task)
            if task_path.exists() and task_path.is_file():
                task = task_path.read_text()

            # Load or use existing configuration
            if self.config is None:
                config = self.load_config(provider, model, api_key, config_file, max_steps)
            else:
                config = self.config

            # Create agent
            agent = self.create_agent(config)

            # Set up trajectory recording
            trajectory_path = None
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

            return {
                "success": True,
                "result": result,
                "trajectory_path": trajectory_path,
                "working_dir": working_dir,
                "config": {
                    "provider": config.default_provider,
                    "model": config.model_providers[config.default_provider].model,
                    "max_steps": config.max_steps,
                },
            }

        except KeyboardInterrupt:
            if verbose:
                console.print("\n[yellow]Task execution interrupted by user[/yellow]")
            return {
                "success": False,
                "error": "Task execution interrupted by user",
                "trajectory_path": trajectory_path if "trajectory_path" in locals() else None,
            }
        except Exception as e:
            if verbose:
                console.print(f"\n[red]Unexpected error: {e}[/red]")
                console.print(traceback.format_exc())
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "trajectory_path": trajectory_path if "trajectory_path" in locals() else None,
            }
        finally:
            # Restore original working directory
            os.chdir(original_cwd)
