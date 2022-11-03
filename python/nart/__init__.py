# Copyright 2022 SenseTime Group Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import coloredlogs, logging

# preset logging style
_level_style = {
    "critical": {"bold": True, "color": "red"},
    "debug": {"color": "blue"},
    "dbg": {"color": "blue"},
    "err": {"color": "red"},
    "info": {},
    "inf": {},
    "notice": {"color": "magenta"},
    "spam": {"color": "green", "faint": True},
    "success": {"bold": True, "color": "green"},
    "verbose": {"color": "blue"},
    "warning": {"color": "yellow"},
    "wrn": {"color": "yellow"},
}

logging.addLevelName(logging.INFO, "INF")
logging.addLevelName(logging.DEBUG, "DBG")
logging.addLevelName(logging.WARNING, "WRN")
logging.addLevelName(logging.CRITICAL, "ERR")
logger = logging.getLogger("nart")
coloredlogs.install(
    fmt="%(levelname)s %(name)s %(asctime)s %(message)s",
    level_styles=_level_style,
    level=logging.DEBUG,
    logger=logger,
)
logger.setLevel(logging.INFO)

try:
    from . import _nart
    from ._nart import __version__
    from .core.art import FakeParade
    from .core.art import FakeTensor
    from .core.art import FakeOp
    from .core.art import Fakes

    def serialize_v1(parade):
        assert isinstance(parade, FakeParade)
        return _nart.serialize_v1(parade.true_obj)

except ImportError as e:
    import warnings

    logger.warn(f"{e}")
    logger.warn(f"can not import `_nart`, can only use nart.tools now.")
