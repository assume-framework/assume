# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

from dataclasses import dataclass, field


@dataclass
class Issue:
    severity: str  # "error" | "warning"
    file: str
    message: str

    def __str__(self) -> str:
        return f"{self.severity.upper():<7} {self.file}: {self.message}"


@dataclass
class ValidationReport:
    """All issues found while verifying one scenario folder."""

    path: str
    cases: list[str] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)

    @property
    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def error(self, file: str, message: str) -> None:
        self.issues.append(Issue("error", file, message))

    def warning(self, file: str, message: str) -> None:
        self.issues.append(Issue("warning", file, message))

    def scope(self, file: str, prefix: str = "") -> "Reporter":
        """Return a reporter for ``file`` which puts ``prefix`` in front of every message."""
        return Reporter(self, file, prefix)

    def summary(self) -> str:
        counts = f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        if self.ok:
            return f"OK: {self.path} can be loaded ({len(self.cases)} study case(s), {counts})"
        return f"FAILED: {self.path} ({counts})"


class Reporter:
    """Reports issues for one file (and, optionally, one study case or section of it)."""

    def __init__(self, report: ValidationReport, file: str, prefix: str = ""):
        self.report = report
        self.file = file
        self.prefix = prefix

    def error(self, message: str) -> None:
        self.report.error(self.file, self.prefix + message)

    def warning(self, message: str) -> None:
        self.report.warning(self.file, self.prefix + message)

    def nested(self, prefix: str) -> "Reporter":
        """Return a reporter for a section of the same file."""
        return Reporter(self.report, self.file, self.prefix + prefix)
