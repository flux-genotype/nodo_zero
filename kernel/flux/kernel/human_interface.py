from abc import ABC, abstractmethod

class HumanInterface(ABC):
    @abstractmethod
    def request_input(self, prompt: str, context: dict = None) -> str:
        pass

    @abstractmethod
    def approve_continue(self, context: dict) -> bool:
        pass

class CLIHumanInterface(HumanInterface):
    def request_input(self, prompt: str, context: dict = None) -> str:
        print(f"\n=== HUMAN INPUT REQUIRED ===")
        if context:
            for k, v in context.items():
                print(f"{k}: {v}")
        return input(prompt + "\n> ")

    def approve_continue(self, context: dict) -> bool:
        print("\n=== APPROVAL REQUIRED ===")
        if context:
            for k, v in context.items():
                print(f"{k}: {v}")
        response = input("Approve and continue? (y/n) > ")
        return response.lower().startswith("y")

class DummyHumanInterface(HumanInterface):
    def request_input(self, prompt: str, context: dict = None) -> str:
        return context.get("ai_response", "") if context else ""

    def approve_continue(self, context: dict) -> bool:
        return True
