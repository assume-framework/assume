# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: AGPL-3.0-or-later

# Assume Exceptions


class AssumeException(Exception):
    pass


class ValidationError(ValueError):
    def __init__(self, message: str, id: str, field: str):
        super().__init__(message)
        self.field = field
        self.id = id


class InvalidTypeException(AssumeException):
    pass
