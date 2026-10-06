# SPDX-FileCopyrightText: ASSUME Developers
#
# SPDX-License-Identifier: MIT

# Assume Exceptions


class AssumeException(Exception):
    pass


class ValidationError(ValueError):
    def __init__(self, message: str, id: str = None, field: str = None):
        super().__init__(message)
        self.field = field
        self.id = id


class InvalidTypeException(AssumeException):
    pass
