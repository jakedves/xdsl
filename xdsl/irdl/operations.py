from __future__ import annotations

import inspect
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import FunctionType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    TypeAlias,
    cast,
    get_args,
    get_origin,
    overload,
)

from typing_extensions import Self, TypeVar, assert_never

from xdsl.ir import (
    Attribute,
    AttributeInvT,
    Block,
    Operation,
    OpResult,
    OpTraits,
    Region,
    SSAValue,
)
from xdsl.traits import OpTrait
from xdsl.utils.classvar import is_const_classvar
from xdsl.utils.exceptions import (
    ParseError,
    PyRDLOpDefinitionError,
    VerifyException,
)
from xdsl.utils.hints import PropertyType, get_type_var_mapping

from .attributes import (  # noqa: TID251
    IRDLAttrConstraint,
    irdl_list_to_attr_constraint,
    irdl_to_attr_constraint,
    range_constr_coercion,
    single_range_constr_coercion,
)
from .constraints import (  # noqa: TID251
    AnyAttr,
    AttrConstraint,
    ConstraintContext,
    ConstraintVar,
    RangeConstraint,
    RangeOf,
)

if TYPE_CHECKING:
    from xdsl.irdl.declarative_assembly_format import CustomDirective
    from xdsl.parser import Parser
    from xdsl.printer import Printer

# pyright: reportMissingParameterType=false, reportUnknownParameterType=false


#   ___                       _   _
#  / _ \ _ __   ___ _ __ __ _| |_(_) ___  _ __
# | | | | '_ \ / _ \ '__/ _` | __| |/ _ \| '_ \
# | |_| | |_) |  __/ | | (_| | |_| | (_) | | | |
#  \___/| .__/ \___|_|  \__,_|\__|_|\___/|_| |_|
#       |_|

IRDLOperationInvT = TypeVar("IRDLOperationInvT", bound="IRDLOperation")
IRDLOperationCovT = TypeVar("IRDLOperationCovT", bound="IRDLOperation", covariant=True)
IRDLOperationContrT = TypeVar(
    "IRDLOperationContrT", bound="IRDLOperation", contravariant=True
)


@dataclass(init=False)
class IRDLOperation(Operation):
    assembly_format: ClassVar[str | None] = None
    custom_directives: ClassVar[tuple[type[CustomDirective], ...]] = ()

    def __init__(
        self: IRDLOperation,
        *,
        operands: (
            Sequence[SSAValue | Operation | Sequence[SSAValue | Operation] | None]
            | None
        ) = None,
        result_types: Sequence[Attribute | Sequence[Attribute] | None] | None = None,
        properties: Mapping[str, Attribute | None] | None = None,
        attributes: Mapping[str, Attribute | None] | None = None,
        successors: Sequence[Block | Sequence[Block] | None] | None = None,
        regions: (
            Sequence[
                Region
                | None
                | Sequence[Operation]
                | Sequence[Block]
                | Sequence[Region | Sequence[Operation] | Sequence[Block]]
            ]
            | None
        ) = None,
    ):
        if operands is None:
            operands = []
        if result_types is None:
            result_types = []
        if properties is None:
            properties = {}
        if attributes is None:
            attributes = {}
        if successors is None:
            successors = []
        if regions is None:
            regions = []
        irdl_op_init(
            self,
            type(self).get_irdl_definition(),
            operands=operands,
            result_types=result_types,
            properties=properties,
            attributes=attributes,
            successors=successors,
            regions=regions,
        )

    def __post_init__(self):
        op_def = self.get_irdl_definition()
        # Fill in default properties
        for prop_name, prop_def in op_def.properties.items():
            if (
                prop_name not in self.properties
                and not isinstance(prop_def, OptionalDef)
                and prop_def.default_value is not None
            ):
                self.properties[prop_name] = prop_def.default_value

        # Fill in default attributes
        for attr_name, attr_def in op_def.attributes.items():
            if (
                attr_name not in self.attributes
                and not isinstance(attr_def, OptionalDef)
                and attr_def.default_value is not None
            ):
                self.attributes[attr_name] = attr_def.default_value

        return super().__post_init__()

    @classmethod
    def build(
        cls,
        *,
        operands: (
            Sequence[SSAValue | Operation | Sequence[SSAValue | Operation] | None]
            | None
        ) = None,
        result_types: Sequence[Attribute | Sequence[Attribute] | None] | None = None,
        attributes: Mapping[str, Attribute | None] | None = None,
        properties: Mapping[str, Attribute | None] | None = None,
        successors: Sequence[Block | Sequence[Block] | None] | None = None,
        regions: (
            Sequence[
                Region
                | None
                | Sequence[Operation]
                | Sequence[Block]
                | Sequence[Region | Sequence[Operation] | Sequence[Block]]
            ]
            | None
        ) = None,
    ) -> Self:
        """Create a new operation using builders."""
        op = cls.__new__(cls)
        IRDLOperation.__init__(
            op,
            operands=operands,
            result_types=result_types,
            properties=properties,
            attributes=attributes,
            successors=successors,
            regions=regions,
        )
        return op

    @classmethod
    def get_irdl_definition(cls) -> OpDef:
        """Get the IRDL operation definition."""
        ...

    def __eq__(self, other: object) -> bool:
        return self is other

    def __hash__(self) -> int:
        return id(self)


@dataclass
class IRDLOption(ABC):
    """Additional option used in IRDL."""


@dataclass
class AttrSizedSegments(IRDLOption, ABC):
    """
    Expect an attribute on the operation that contains the segment sizes of the
    operand, result, region, or successor lists.
    For instance, the list `[a, b, c, d]` with segment sizes `[1, 3]` will result
    in the `[a], [b, c, d]` lists.
    The attribute must be a dense array of `i32`, its lenght must be equal to the
    number of segments (e.g. the number of operand definitions), and its sum must
    be equal to the number of elements in the list (e.g. the number of operands).
    """

    attribute_name: ClassVar[str]
    as_property: bool = False
    """Name of the attribute containing the segment sizes."""


@dataclass
class AttrSizedOperandSegments(AttrSizedSegments):
    """
    Expect an attribute on the operation that contains the sizes of the operand
    definitions.
    See `AttrSizedSegments` for more information.
    """

    attribute_name = "operandSegmentSizes"
    """Name of the attribute containing the variadic operand sizes."""


@dataclass
class AttrSizedResultSegments(AttrSizedSegments):
    """
    Expect an attribute on the operation that contains the sizes of the result
    definitions.
    See `AttrSizedSegments` for more information.
    """

    attribute_name = "resultSegmentSizes"
    """Name of the attribute containing the variadic result sizes."""


@dataclass
class AttrSizedRegionSegments(AttrSizedSegments):
    """
    Expect an attribute on the operation that contains the sizes of the region
    definitions.
    See `AttrSizedSegments` for more information.
    """

    attribute_name = "regionSegmentSizes"
    """Name of the attribute containing the variadic region sizes."""


@dataclass
class AttrSizedSuccessorSegments(AttrSizedSegments):
    """
    Expect an attribute on the operation that contains the sizes of the successor
    definitions.
    See `AttrSizedSegments` for more information.
    """

    attribute_name = "successorSegmentSizes"
    """Name of the attribute containing the variadic successor sizes."""


class SameVariadicSize(IRDLOption):
    """
    All variadic definitions should have the same size.
    """


class SameVariadicResultSize(SameVariadicSize):
    """
    All variadic results should have the same size.
    """


class SameVariadicOperandSize(SameVariadicSize):
    """
    All variadic operands should have the same size.
    """


class SameVariadicRegionSize(SameVariadicSize):
    """
    All variadic regions should have the same size.
    """


class SameVariadicSuccessorSize(SameVariadicSize):
    """
    All variadic successors should have the same size.
    """


@dataclass
class ParsePropInAttrDict(IRDLOption):
    """
    Allows properties to be omitted from the assembly format, causing them
    to be parsed as part of the attribute dictionary.
    This should only be used to ensure MLIR compatibility, it is otherwise
    bad design to use it.
    """


@dataclass
class OperandOrResultDef(ABC):
    """An operand or a result definition. Should not be used directly."""


@dataclass
class VariadicDef(OperandOrResultDef):
    """A variadic operand or result definition. Should not be used directly."""


@dataclass
class OptionalDef(VariadicDef):
    """An optional operand or result definition. Should not be used directly."""


@dataclass(init=False)
class OperandDef(OperandOrResultDef):
    """An IRDL operand definition."""

    constr: RangeConstraint
    """The operand constraint."""

    def __init__(self, attr: Attribute | type[Attribute] | AttrConstraint):
        self.constr = single_range_constr_coercion(attr)


Operand: TypeAlias = SSAValue


@dataclass(init=False)
class VarOperandDef(OperandDef, VariadicDef):
    """An IRDL variadic operand definition."""

    def __init__(
        self,
        attr: Attribute | type[Attribute] | AttrConstraint | RangeConstraint,
    ):
        self.constr = range_constr_coercion(attr)


class VarOperand(tuple[Operand, ...]):
    @property
    def types(self):
        return tuple(o.type for o in self)


@dataclass(init=False)
class OptOperandDef(VarOperandDef, OptionalDef):
    """An IRDL optional operand definition."""


OptOperand: TypeAlias = SSAValue | None


@dataclass(init=False)
class ResultDef(OperandOrResultDef):
    """An IRDL result definition."""

    constr: RangeConstraint
    """The result constraint."""

    def __init__(
        self, attr: Attribute | type[Attribute] | AttrConstraint | RangeConstraint
    ):
        assert not isinstance(attr, RangeConstraint)
        self.constr = single_range_constr_coercion(attr)


@dataclass(init=False)
class VarResultDef(ResultDef, VariadicDef):
    """An IRDL variadic result definition."""

    def __init__(
        self, attr: Attribute | type[Attribute] | AttrConstraint | RangeConstraint
    ):
        self.constr = range_constr_coercion(attr)


class VarOpResult(Generic[AttributeInvT], tuple[OpResult[AttributeInvT], ...]):
    @property
    def types(self):
        return tuple(r.type for r in self)


@dataclass(init=False)
class OptResultDef(VarResultDef, OptionalDef):
    """An IRDL optional result definition."""


OptOpResult: TypeAlias = OpResult[AttributeInvT] | None


@dataclass(init=True)
class RegionDef:
    """
    An IRDL region definition.
    """

    entry_args: RangeConstraint = field(default_factory=lambda: RangeOf(AnyAttr()))


@dataclass
class VarRegionDef(RegionDef, VariadicDef):
    """An IRDL variadic region definition."""


@dataclass
class OptRegionDef(RegionDef, OptionalDef):
    """An IRDL optional region definition."""


VarRegion: TypeAlias = tuple[Region, ...]
OptRegion: TypeAlias = Region | None


@dataclass
class SingleBlockRegionDef(RegionDef):
    """An IRDL region definition that expects exactly one block."""


class VarSingleBlockRegionDef(RegionDef, VariadicDef):
    """An IRDL variadic region definition that expects exactly one block."""


class OptSingleBlockRegionDef(RegionDef, OptionalDef):
    """An IRDL optional region definition that expects exactly one block."""


@dataclass
class AttrOrPropDef:
    """An IRDL attribute or property definition."""

    constr: AttrConstraint
    default_value: Attribute | None = None


@dataclass
class AttributeDef(AttrOrPropDef):
    """An IRDL attribute definition."""


@dataclass
class OptAttributeDef(AttributeDef, OptionalDef):
    """An IRDL attribute definition for an optional attribute."""


@dataclass
class PropertyDef(AttrOrPropDef):
    """An IRDL property definition."""


@dataclass
class OptPropertyDef(PropertyDef, OptionalDef):
    """An IRDL property definition for an optional property."""


class SuccessorDef:
    """An IRDL successor definition."""


class VarSuccessorDef(SuccessorDef, VariadicDef):
    """An IRDL variadic successor definition."""


class OptSuccessorDef(SuccessorDef, OptionalDef):
    """An IRDL optional successor definition."""


Successor: TypeAlias = Block
OptSuccessor: TypeAlias = Block | None
VarSuccessor: TypeAlias = list[Block]

_ClsT = TypeVar("_ClsT")

# Field definition classes for `@irdl_op_definition`
# They carry the type information exactly as passed in the argument to `operand_def` etc.
# We can only convert them to constraints when creating the OpDef to allow for type var
# mapping.


class _OpDefField(Generic[_ClsT]):
    cls: type[_ClsT]

    def __init__(self, cls: type[_ClsT]):
        self.cls = cls


class _RangeConstrainedOpDefField(Generic[_ClsT], _OpDefField[_ClsT]):
    param: RangeConstraint | IRDLAttrConstraint

    def __init__(self, cls: type[_ClsT], param: RangeConstraint | IRDLAttrConstraint):
        super().__init__(cls)
        self.param = param


class _ConstrainedOpDefField(Generic[_ClsT], _OpDefField[_ClsT]):
    param: IRDLAttrConstraint

    def __init__(self, cls: type[_ClsT], param: IRDLAttrConstraint):
        super().__init__(cls)
        self.param = param


class _OperandFieldDef(_RangeConstrainedOpDefField[OperandDef,]):
    pass


class _ResultFieldDef(_RangeConstrainedOpDefField[ResultDef]):
    pass


AttrOrPropInvT = TypeVar("AttrOrPropInvT", bound=AttrOrPropDef)


class _AttrOrPropFieldDef(
    Generic[AttrOrPropInvT], _ConstrainedOpDefField[AttrOrPropInvT]
):
    ir_name: str | None = None
    """
    The name of the attribute or property in the IR,
    in case it is different from the field name.
    """
    default_value: Attribute | None = None

    def __init__(
        self,
        cls: type[AttrOrPropInvT],
        param: IRDLAttrConstraint,
        ir_name: str | None = None,
        default_value: Attribute | None = None,
    ):
        super().__init__(cls, param)
        self.ir_name = ir_name
        self.default_value = default_value


class _AttributeFieldDef(_AttrOrPropFieldDef[AttributeDef]):
    pass


class _PropertyFieldDef(_AttrOrPropFieldDef[PropertyDef]):
    pass


class _RegionFieldDef(_OpDefField[RegionDef]):
    entry_args: RangeConstraint | IRDLAttrConstraint

    def __init__(
        self,
        cls: type[RegionDef],
        entry_args: RangeConstraint | IRDLAttrConstraint,
    ):
        super().__init__(cls)
        self.entry_args = entry_args

    pass


class _SuccessorFieldDef(_OpDefField[SuccessorDef]):
    pass


def result_def(
    constraint: IRDLAttrConstraint[AttributeInvT] = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OpResult[AttributeInvT]:
    """
    Defines a result of an operation.
    """
    return cast(OpResult[AttributeInvT], _ResultFieldDef(ResultDef, constraint))


def var_result_def(
    constraint: (
        RangeConstraint[AttributeInvT] | IRDLAttrConstraint[AttributeInvT]
    ) = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarOpResult[AttributeInvT]:
    """
    Defines a variadic result of an operation.
    """
    return cast(VarOpResult[AttributeInvT], _ResultFieldDef(VarResultDef, constraint))


def opt_result_def(
    constraint: (
        RangeConstraint[AttributeInvT] | IRDLAttrConstraint[AttributeInvT]
    ) = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptOpResult[AttributeInvT]:
    """
    Defines an optional result of an operation.
    """
    return cast(OptOpResult[AttributeInvT], _ResultFieldDef(OptResultDef, constraint))


def prop_def(
    constraint: IRDLAttrConstraint[AttributeInvT] = Attribute,
    default_value: Attribute | None = None,
    *,
    prop_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT:
    """Defines a property of an operation."""
    return cast(
        AttributeInvT,
        _PropertyFieldDef(PropertyDef, constraint, prop_name, default_value),
    )


@overload
def opt_prop_def(
    constraint: IRDLAttrConstraint[AttributeInvT],
    default_value: Attribute,
    *,
    prop_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT: ...


@overload
def opt_prop_def(
    constraint: IRDLAttrConstraint[AttributeInvT] = Attribute,
    default_value: Attribute | None = None,
    *,
    prop_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT | None: ...


def opt_prop_def(
    constraint: IRDLAttrConstraint[AttributeInvT] = Attribute,
    default_value: Attribute | None = None,
    *,
    prop_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT | None:
    """Defines an optional property of an operation."""
    return cast(
        AttributeInvT,
        _PropertyFieldDef(OptPropertyDef, constraint, prop_name, default_value),
    )


def attr_def(
    constraint: IRDLAttrConstraint[AttributeInvT] = Attribute,
    default_value: Attribute | None = None,
    *,
    attr_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT:
    """
    Defines an attribute of an operation.
    """
    return cast(
        AttributeInvT,
        _AttributeFieldDef(AttributeDef, constraint, attr_name, default_value),
    )


@overload
def opt_attr_def(
    constraint: IRDLAttrConstraint[AttributeInvT],
    default_value: Attribute,
    *,
    attr_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT: ...


@overload
def opt_attr_def(
    constraint: IRDLAttrConstraint[AttributeInvT] = Attribute,
    default_value: Attribute | None = None,
    *,
    attr_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT | None: ...


def opt_attr_def(
    constraint: IRDLAttrConstraint[AttributeInvT] = Attribute,
    default_value: Attribute | None = None,
    *,
    attr_name: str | None = None,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> AttributeInvT | None:
    """
    Defines an optional attribute of an operation.
    """
    return cast(
        AttributeInvT,
        _AttributeFieldDef(OptAttributeDef, constraint, attr_name, default_value),
    )


def operand_def(
    constraint: IRDLAttrConstraint = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> Operand:
    """
    Defines an operand of an operation.
    """
    return cast(Operand, _OperandFieldDef(OperandDef, constraint))


def var_operand_def(
    constraint: RangeConstraint | IRDLAttrConstraint = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarOperand:
    """
    Defines a variadic operand of an operation.
    """
    return cast(VarOperand, _OperandFieldDef(VarOperandDef, constraint))


def opt_operand_def(
    constraint: RangeConstraint | IRDLAttrConstraint = Attribute,
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptOperand:
    """
    Defines an optional operand of an operation.
    """
    return cast(OptOperand, _OperandFieldDef(OptOperandDef, constraint))


def region_def(
    single_block: Literal["single_block"] | None = None,
    *,
    entry_args: RangeConstraint | IRDLAttrConstraint = RangeOf(AnyAttr()),
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> Region:
    """
    Defines a region of an operation.
    """
    cls = RegionDef if single_block is None else SingleBlockRegionDef
    return cast(Region, _RegionFieldDef(cls, entry_args))


def var_region_def(
    single_block: Literal["single_block"] | None = None,
    *,
    entry_args: RangeConstraint | IRDLAttrConstraint = RangeOf(AnyAttr()),
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarRegion:
    """
    Defines a variadic region of an operation.
    """
    cls = VarRegionDef if single_block is None else VarSingleBlockRegionDef
    return cast(VarRegion, _RegionFieldDef(cls, entry_args))


def opt_region_def(
    single_block: Literal["single_block"] | None = None,
    *,
    entry_args: RangeConstraint | IRDLAttrConstraint = RangeOf(AnyAttr()),
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptRegion:
    """
    Defines an optional region of an operation.
    """
    cls = OptRegionDef if single_block is None else OptSingleBlockRegionDef
    return cast(OptRegion, _RegionFieldDef(cls, entry_args))


def successor_def(
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> Successor:
    """
    Defines a successor of an operation.
    """
    return cast(Successor, _SuccessorFieldDef(SuccessorDef))


def var_successor_def(
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> VarSuccessor:
    """
    Defines a variadic successor of an operation.
    """
    return cast(VarSuccessor, _SuccessorFieldDef(VarSuccessorDef))


def opt_successor_def(
    *,
    default: None = None,
    resolver: None = None,
    init: Literal[False] = False,
) -> OptSuccessor:
    """
    Defines an optional successor of an operation.
    """
    return cast(OptSuccessor, _SuccessorFieldDef(OptSuccessorDef))


# traits


def traits_def(*traits: OpTrait):
    """
    Defines the traits of an operation.
    """
    return OpTraits(frozenset(traits))


def lazy_traits_def(future_traits: Callable[[], tuple[OpTrait, ...]]):
    """
    Defines the traits of an operation, in the case where any trait uses an operation
    that is not yet declared.
    """
    return OpTraits(future_traits)


# Exclude `object`
_OPERATION_DICT_KEYS = {key for cls in Operation.mro()[:-1] for key in cls.__dict__}


@dataclass(kw_only=True)
class OpDef:
    """The internal IRDL definition of an operation."""

    name: str = field(kw_only=False)
    operands: list[tuple[str, OperandDef]] = field(
        default_factory=list[tuple[str, OperandDef]]
    )
    results: list[tuple[str, ResultDef]] = field(
        default_factory=list[tuple[str, ResultDef]]
    )
    properties: dict[str, PropertyDef] = field(default_factory=dict[str, PropertyDef])
    attributes: dict[str, AttributeDef] = field(default_factory=dict[str, AttributeDef])
    regions: list[tuple[str, RegionDef]] = field(
        default_factory=list[tuple[str, RegionDef]]
    )
    successors: list[tuple[str, SuccessorDef]] = field(
        default_factory=list[tuple[str, SuccessorDef]]
    )
    options: list[IRDLOption] = field(default_factory=list[IRDLOption])
    traits: OpTraits = field(default_factory=lambda: traits_def())

    accessor_names: dict[str, tuple[str, Literal["attribute", "property"]]] = field(
        default_factory=dict[str, tuple[str, Literal["attribute", "property"]]]
    )
    """
    Mapping from the accessor name to the attribute or property name.
    In some cases, the attribute name is not a valid Python identifier,
    or is already used by the operation, so we need to use a different name.
    """
    assembly_format: str | None = field(default=None)
    custom_directives: dict[str, type[CustomDirective]] = field(
        default_factory=lambda: {}
    )

    @staticmethod
    def from_pyrdl(pyrdl_def: type[IRDLOperationInvT]) -> OpDef:
        """Decorator used on classes to define a new operation definition."""

        type_var_mapping: dict[TypeVar, AttrConstraint] | None = None

        # If the operation inherit from `Generic`, this means that it specializes a
        # generic operation. Retrieve the mapping from `TypeVar` to pyrdl constraints.
        if issubclass(pyrdl_def, Generic):
            type_var_mapping = {
                k: irdl_to_attr_constraint(v)
                for k, v in get_type_var_mapping(pyrdl_def)[1].items()
            }

        def wrong_field_exception(field_name: str) -> PyRDLOpDefinitionError:
            raise PyRDLOpDefinitionError(
                f"{pyrdl_def.__name__}.{field_name} is neither a function,"
                "operand, result, region, or attribute definition. "
                "Operands should be defined with type hints of "
                "operand_def(<Constraint>), results with "
                "result_def(<Constraint>), regions with "
                "region_def(), attributes with "
                "attr_def(<Constraint>), properties with prop_def(<Constraint>), "
                "and constants (indicated by uppercase field names) as ClassVar."
            )

        op_def = OpDef(pyrdl_def.name)

        # If an operation subclass overrides a superclass field, only keep the definition
        # of the subclass.
        field_names = set[str]()

        # Get all fields of the class, including the parent classes
        for parent_cls in pyrdl_def.mro():
            # Do not collect fields from Generic, as Generic will not contain
            # IRDL definitions, and contains ClassVar fields that are not
            # allowed in IRDL definitions.
            if parent_cls == Generic:
                continue
            if parent_cls in Operation.mro():
                continue

            clsdict = parent_cls.__dict__

            annotations = parent_cls.__annotations__

            for field_name in annotations:
                if field_name not in clsdict:
                    if is_const_classvar(
                        field_name, annotations[field_name], PyRDLOpDefinitionError
                    ):
                        continue
                    raise wrong_field_exception(field_name)

            for field_name in clsdict:
                if field_name in ("name", "assembly_format", "custom_directives"):
                    continue
                if field_name in _OPERATION_DICT_KEYS:
                    # Fields that are already in Operation (i.e. operands, results, ...)
                    continue
                if field_name in field_names:
                    # already registered value for field name
                    continue
                if field_name in annotations and is_const_classvar(
                    field_name, annotations[field_name], PyRDLOpDefinitionError
                ):
                    continue

                value = clsdict[field_name]

                # Check that all fields of the operation definition are either already
                # in Operation, or are class functions or methods.

                if field_name == "irdl_options":
                    value = cast(list[IRDLOption], value)
                    op_def.options.extend(value)
                    for option in value:
                        if isinstance(option, AttrSizedSegments):
                            defs = (
                                op_def.properties
                                if option.as_property
                                else op_def.attributes
                            )
                            def_name = "property" if option.as_property else "attribute"
                            if option.attribute_name in defs:
                                raise PyRDLOpDefinitionError(
                                    f"pyrdl operation definition '{pyrdl_def.__name__}' "
                                    f"has a '{option.attribute_name}' {def_name}, which "
                                    "is incompatible with the "
                                    f"{option} option."
                                )
                            from xdsl.dialects.builtin import DenseArrayBase

                            if option.as_property:
                                prop_def = PropertyDef(
                                    irdl_to_attr_constraint(DenseArrayBase)
                                )
                                op_def.properties[option.attribute_name] = prop_def
                            else:
                                attr_def = AttributeDef(
                                    irdl_to_attr_constraint(DenseArrayBase)
                                )
                                op_def.attributes[option.attribute_name] = attr_def
                    continue

                if field_name == "traits":
                    traits = value
                    field_names.add("traits")
                    if not isinstance(traits, OpTraits):
                        raise PyRDLOpDefinitionError(
                            f"pyrdl operation definition '{pyrdl_def.__name__}' "
                            "traits field should be an instance of"
                            f"'{OpTraits.__name__}'."
                        )
                    op_def.traits = traits
                    continue

                # Dunder fields are allowed (i.e. __orig_bases__, __annotations__, ...)
                # They are used by Python to store information about the class, so they
                # should not be considered as part of the operation definition.
                # Also, they can provide a possiblea escape hatch.
                if field_name[:2] == "__" and field_name[-2:] == "__":
                    continue

                # Methods, properties, and functions are allowed
                if isinstance(
                    value, FunctionType | PropertyType | classmethod | staticmethod
                ):
                    continue
                # Constraint variables are deprecated
                if get_origin(value) is Annotated:
                    if any(isinstance(arg, ConstraintVar) for arg in get_args(value)):
                        import warnings

                        warnings.warn(
                            "The use of `ConstraintVar` is deprecated, please use `VarConstraint`",
                            DeprecationWarning,
                            stacklevel=2,
                        )
                        continue

                # Get attribute constraints from a list of pyrdl constraints
                def get_constraint(
                    pyrdl_constr: IRDLAttrConstraint,
                ) -> AttrConstraint:
                    constraint = irdl_list_to_attr_constraint(
                        (pyrdl_constr,),
                        allow_type_var=True,
                    )
                    if type_var_mapping is not None:
                        constraint = constraint.mapping_type_vars(type_var_mapping)
                    return constraint

                # Get attribute constraints from a list of pyrdl constraints
                def get_range_constraint(
                    pyrdl_constr: RangeConstraint | IRDLAttrConstraint,
                ) -> RangeConstraint:
                    if isinstance(pyrdl_constr, RangeConstraint):
                        # Pyright does not know the type of the generic range constraint
                        return cast(RangeConstraint, pyrdl_constr)
                    return RangeOf(get_constraint(pyrdl_constr))

                field_names.add(field_name)

                match value:
                    case _ResultFieldDef():
                        if not issubclass(value.cls, VariadicDef):
                            if isinstance(value.param, RangeConstraint):
                                raise TypeError(
                                    "Cannot use a RangeConstraint in result_def, use an "
                                    "AttrConstraint or var_result_def or "
                                    "opt_result_def instead."
                                )
                            constraint = get_constraint(value.param)
                            result_def = value.cls(constraint)
                        else:
                            constraint = get_range_constraint(value.param)
                            result_def = value.cls(constraint)
                        op_def.results.append((field_name, result_def))
                        continue
                    case _OperandFieldDef():
                        if not issubclass(value.cls, VariadicDef):
                            if isinstance(value.param, RangeConstraint):
                                raise TypeError(
                                    "Cannot use a RangeConstraint in operand_def, use an "
                                    "AttrConstraint or var_operand_def or "
                                    "opt_operand_def instead."
                                )
                            constraint = get_constraint(value.param)
                            operand_def = value.cls(constraint)
                        else:
                            constraint = get_range_constraint(value.param)
                            operand_def = cast(type[VarOperandDef], value.cls)(
                                constraint
                            )
                        op_def.operands.append((field_name, operand_def))
                        continue
                    case _AttributeFieldDef():
                        constraint = get_constraint(value.param)
                        attribute_def = value.cls(constraint, value.default_value)
                        ir_name = field_name if value.ir_name is None else value.ir_name
                        op_def.attributes[ir_name] = attribute_def
                        op_def.accessor_names[field_name] = (ir_name, "attribute")
                        continue
                    case _PropertyFieldDef():
                        constraint = get_constraint(value.param)
                        property_def = value.cls(constraint, value.default_value)
                        ir_name = field_name if value.ir_name is None else value.ir_name
                        op_def.properties[ir_name] = property_def
                        op_def.accessor_names[field_name] = (ir_name, "property")
                        continue
                    case _RegionFieldDef():
                        constraint = get_range_constraint(value.entry_args)
                        region_def = value.cls(constraint)
                        op_def.regions.append((field_name, region_def))
                        continue
                    case _SuccessorFieldDef():
                        successor_def = value.cls()
                        op_def.successors.append((field_name, successor_def))
                        continue
                    case _:
                        pass

                raise wrong_field_exception(field_name)

        op_def.assembly_format = pyrdl_def.assembly_format
        op_def.custom_directives = {
            directive.__name__: directive for directive in pyrdl_def.custom_directives
        }
        assert inspect.ismethod(Operation.parse)
        if op_def.assembly_format is not None and (
            pyrdl_def.print != Operation.print
            or not inspect.ismethod(pyrdl_def.parse)
            or pyrdl_def.parse.__func__ != Operation.parse.__func__
        ):
            raise PyRDLOpDefinitionError(
                "Cannot define both an assembly format (with the assembly_format "
                "variable) and the print and parse methods."
            )

        return op_def

    def verify(self, op: Operation):
        """Given an IRDL definition, verify that an operation satisfies its invariants."""

        # Mapping from type variables to their concrete types.
        constraint_context = ConstraintContext()

        # Verify operands.
        irdl_op_verify_arg_list(op, self, VarIRConstruct.OPERAND, constraint_context)

        # Verify results.
        irdl_op_verify_arg_list(op, self, VarIRConstruct.RESULT, constraint_context)

        # Verify regions.
        irdl_op_verify_regions(op, self, constraint_context)

        # Verify successors.
        get_variadic_sizes(op, self, VarIRConstruct.SUCCESSOR)

        # Verify properties.
        for prop_name, attr_def in self.properties.items():
            if prop_name not in op.properties:
                if isinstance(attr_def, OptPropertyDef):
                    continue
                raise VerifyException(
                    f"property '{prop_name}' expected in operation '{op.name}'"
                )
            attr_def.constr.verify(op.properties[prop_name], constraint_context)

        for prop_name in op.properties.keys():
            if prop_name not in self.properties:
                raise VerifyException(
                    f"property '{prop_name}' is not defined by the operation '{op.name}'. "
                    "Use the dictionary attribute to add arbitrary information "
                    "to the operation."
                )

        # Verify attributes.
        for attr_name, attr_def in self.attributes.items():
            if attr_name not in op.attributes:
                if isinstance(attr_def, OptAttributeDef):
                    continue
                raise VerifyException(
                    f"attribute '{attr_name}' expected in operation '{op.name}'"
                )
            attr_def.constr.verify(op.attributes[attr_name], constraint_context)

        # Verify traits.
        for trait in self.traits:
            trait.verify(op)


class VarIRConstruct(Enum):
    """
    An enum representing the part of an IR that may be variadic.
    This contains operands, results, and regions.
    """

    OPERAND = 1
    RESULT = 2
    REGION = 3
    SUCCESSOR = 4


def get_construct_name(construct: VarIRConstruct) -> str:
    """Get the type name, this is used mostly for error messages."""
    match construct:
        case VarIRConstruct.OPERAND:
            return "operand"
        case VarIRConstruct.RESULT:
            return "result"
        case VarIRConstruct.REGION:
            return "region"
        case VarIRConstruct.SUCCESSOR:
            return "successor"


def get_construct_defs(
    op_def: OpDef, construct: VarIRConstruct
) -> (
    list[tuple[str, OperandDef]]
    | list[tuple[str, ResultDef]]
    | list[tuple[str, RegionDef]]
    | list[tuple[str, SuccessorDef]]
):
    """Get the definitions of this type in an operation definition."""
    match construct:
        case VarIRConstruct.OPERAND:
            return op_def.operands
        case VarIRConstruct.RESULT:
            return op_def.results
        case VarIRConstruct.REGION:
            return op_def.regions
        case VarIRConstruct.SUCCESSOR:
            return op_def.successors
    assert_never(construct)


def get_op_constructs(
    op: Operation, construct: VarIRConstruct
) -> Sequence[SSAValue] | Sequence[OpResult] | Sequence[Region] | Sequence[Successor]:
    """
    Get the list of arguments of the type in an operation.
    For example, if the argument type is an operand, get the list of
    operands.
    """
    match construct:
        case VarIRConstruct.OPERAND:
            return op.operands
        case VarIRConstruct.RESULT:
            return op.results
        case VarIRConstruct.REGION:
            return op.regions
        case VarIRConstruct.SUCCESSOR:
            return op.successors
    assert_never(construct)


def get_attr_size_option(
    construct: VarIRConstruct,
) -> type[
    AttrSizedOperandSegments
    | AttrSizedResultSegments
    | AttrSizedRegionSegments
    | AttrSizedSuccessorSegments
]:
    """Get the AttrSized option for this type."""
    match construct:
        case VarIRConstruct.OPERAND:
            return AttrSizedOperandSegments
        case VarIRConstruct.RESULT:
            return AttrSizedResultSegments
        case VarIRConstruct.REGION:
            return AttrSizedRegionSegments
        case VarIRConstruct.SUCCESSOR:
            return AttrSizedSuccessorSegments
    assert_never(construct)


def get_same_variadic_size_option(
    construct: VarIRConstruct,
) -> type[
    SameVariadicOperandSize
    | SameVariadicResultSize
    | SameVariadicRegionSize
    | SameVariadicSuccessorSize
]:
    """Get the AttrSized option for this type."""
    match construct:
        case VarIRConstruct.OPERAND:
            return SameVariadicOperandSize
        case VarIRConstruct.RESULT:
            return SameVariadicResultSize
        case VarIRConstruct.REGION:
            return SameVariadicRegionSize
        case VarIRConstruct.SUCCESSOR:
            return SameVariadicSuccessorSize
    assert_never(construct)


def get_multiple_variadic_options(
    construct: VarIRConstruct,
) -> list[type[IRDLOption]]:
    return [get_same_variadic_size_option(construct), get_attr_size_option(construct)]


def get_variadic_sizes_from_attr(
    op: Operation,
    defs: Sequence[tuple[str, OperandDef | ResultDef | RegionDef | SuccessorDef]],
    construct: VarIRConstruct,
    size_attribute_name: str,
    from_prop: bool = False,
) -> list[int]:
    """
    Get the sizes of the variadic definitions
    from the corresponding attribute.
    """
    # Circular import because DenseArrayBase is defined using IRDL
    from xdsl.dialects.builtin import DenseArrayBase, i32

    container = op.properties if from_prop else op.attributes
    container_name = "property" if from_prop else "attribute"

    # Check that the attribute is present
    if size_attribute_name not in container:
        raise VerifyException(
            f"Expected {size_attribute_name} {container_name} in {op.name} operation."
        )
    attribute = container[size_attribute_name]
    if not isinstance(attribute, DenseArrayBase) or attribute.elt_type != i32:  # pyright: ignore[reportUnknownMemberType]
        raise VerifyException(
            f"{size_attribute_name} {container_name} is expected "
            "to be a DenseArrayBase of i32."
        )

    def_sizes = attribute.get_values()

    if len(def_sizes) != len(defs):
        raise VerifyException(
            f"expected {len(defs)} values in "
            f"{size_attribute_name}, but got {len(def_sizes)}"
        )

    variadic_sizes = list[int]()
    for (arg_name, arg_def), arg_size in zip(defs, def_sizes):
        if isinstance(arg_def, OptionalDef) and arg_size > 1:
            raise VerifyException(
                f"optional {get_construct_name(construct)} {arg_name} is expected to "
                f"be of size 0 or 1 in {size_attribute_name}, but got "
                f"{arg_size}"
            )

        if not isinstance(arg_def, VariadicDef) and arg_size != 1:
            raise VerifyException(
                f"non-variadic {get_construct_name(construct)} {arg_name} is expected "
                f"to be of size 1 in {size_attribute_name}, but got {arg_size}"
            )

        if isinstance(arg_def, VariadicDef):
            variadic_sizes.append(arg_size)

    return variadic_sizes


def get_variadic_sizes(
    op: Operation, op_def: OpDef, construct: VarIRConstruct
) -> list[int]:
    """Get variadic sizes of operands or results."""

    defs = get_construct_defs(op_def, construct)
    args = get_op_constructs(op, construct)
    def_type_name = get_construct_name(construct)
    attribute_option = get_attr_size_option(construct)
    same_size_option = get_same_variadic_size_option(construct)

    variadic_defs = [
        (arg_name, arg_def)
        for arg_name, arg_def in defs
        if isinstance(arg_def, VariadicDef)
    ]

    # If the size is in the attributes, fetch it
    option = next((o for o in op_def.options if isinstance(o, attribute_option)), None)
    if option is not None:
        return get_variadic_sizes_from_attr(
            op,
            defs,
            construct,
            option.attribute_name,
            option.as_property,
        )

    # If there are no variadics arguments,
    # we just check that we have the right number of arguments
    if len(variadic_defs) == 0:
        if len(args) != len(defs):
            raise VerifyException(
                f"Expected {len(defs)} {def_type_name}, but got {len(args)}"
            )
        return []

    # If there is a single variadic argument,
    # we can get its size from the number of arguments.
    if len(variadic_defs) == 1:
        if len(args) - len(defs) + 1 < 0:
            raise VerifyException(
                f"Expected at least {len(defs) - 1} {def_type_name}s, got {len(defs)}"
            )
        return [len(args) - len(defs) + 1]

    # If the operation has to related SameSize option, equally distribute the
    # variadic arguments between the variadic definitions.
    option = next((o for o in op_def.options if isinstance(o, same_size_option)), None)

    assert option is not None, "Unexpected xDSL error while fetching variadic sizes"

    non_variadic_defs = len(defs) - len(variadic_defs)
    variadic_args = len(args) - non_variadic_defs
    if variadic_args % len(variadic_defs):
        name = get_construct_name(construct)
        raise VerifyException(
            f"Operation has {variadic_args} {name}s for {len(variadic_defs)} variadic {name}s marked as having the same size."
        )
    return [variadic_args // len(variadic_defs)] * len(variadic_defs)


def get_operand_result_or_region(
    op: Operation,
    op_def: OpDef,
    arg_def_idx: int,
    previous_var_args: int,
    construct: VarIRConstruct,
) -> (
    None
    | Operand
    | VarOperand
    | OptOperand
    | OpResult
    | VarOpResult
    | OptOpResult
    | Sequence[SSAValue]
    | Sequence[OpResult]
    | Region
    | Sequence[Region]
    | Successor
    | Sequence[Successor]
):
    """
    Get an operand, result, or region.
    In the case of a variadic definition, return a list of elements.
    :param op: The operation we want to get argument of.
    :param arg_def_idx: The index of the argument in the irdl definition.
    :param previous_var_args: The number of previous variadic definitions
           before this definition.
    :param arg_type: The type of the argument we want
           (i.e. operand, result, or region)
    :return:
    """
    defs = get_construct_defs(op_def, construct)
    args = get_op_constructs(op, construct)

    variadic_sizes = get_variadic_sizes(op, op_def, construct)

    begin_arg = (
        arg_def_idx - previous_var_args + sum(variadic_sizes[:previous_var_args])
    )
    if isinstance(defs[arg_def_idx][1], OptionalDef):
        arg_size = variadic_sizes[previous_var_args]
        if arg_size == 0:
            return None
        else:
            return args[begin_arg]
    if isinstance(defs[arg_def_idx][1], VariadicDef):
        arg_size = variadic_sizes[previous_var_args]
        values = args[begin_arg : begin_arg + arg_size]
        if isinstance(defs[arg_def_idx][1], OperandDef):
            return VarOperand(cast(Sequence[Operand], values))
        elif isinstance(defs[arg_def_idx][1], ResultDef):
            return VarOpResult(cast(Sequence[OpResult], values))
        else:
            return values
    else:
        return args[begin_arg]


def irdl_op_verify_regions(
    op: Operation, op_def: OpDef, constraint_context: ConstraintContext
):
    get_variadic_sizes(op, op_def, VarIRConstruct.REGION)
    for i, (region, (name, region_def)) in enumerate(zip(op.regions, op_def.regions)):
        if isinstance(region_def, SingleBlockRegionDef) and len(region.blocks) != 1:
            raise VerifyException(
                f"Region '{name}' at position {i} expected a single block, but got "
                f"{len(region.blocks)} blocks"
            )
        if (first_block := region.blocks.first) is not None:
            entry_args_types = first_block.arg_types
            try:
                region_def.entry_args.verify(entry_args_types, constraint_context)
            except VerifyException as e:
                raise VerifyException(
                    f"region #{i} entry arguments do not verify:\n{e}"
                ) from e


def irdl_op_verify_arg_list(
    op: Operation,
    op_def: OpDef,
    construct: Literal[VarIRConstruct.OPERAND, VarIRConstruct.RESULT],
    constraint_context: ConstraintContext,
) -> None:
    """Verify the argument list of an operation."""
    arg_sizes = get_variadic_sizes(op, op_def, construct)
    arg_idx = 0
    var_idx = 0
    args = cast(
        Sequence[SSAValue] | Sequence[OpResult], get_op_constructs(op, construct)
    )
    args_defs = cast(
        Sequence[tuple[str, ResultDef]] | Sequence[tuple[str, OperandDef]],
        get_construct_defs(op_def, construct),
    )

    def verify_sequence(
        arg: Sequence[SSAValue], arg_def: ResultDef | OperandDef, arg_idx: int
    ) -> None:
        """Verify a single argument."""
        try:
            arg_def.constr.verify(tuple(a.type for a in arg), constraint_context)
        except VerifyException as e:
            if len(arg) == 1:
                pos = f"{arg_idx}"
            else:
                pos = f"{arg_idx} to {arg_idx + len(arg) - 1}"
            raise VerifyException(
                f"{get_construct_name(construct)} at position {pos} does not "
                f"verify:\n{e}"
            ) from e

    for def_idx, (_, arg_def) in enumerate(args_defs):
        if isinstance(arg_def, VariadicDef):
            verify_sequence(
                args[arg_idx : arg_idx + arg_sizes[var_idx]], arg_def, def_idx
            )
            arg_idx += arg_sizes[var_idx]
            var_idx += 1
        else:
            verify_sequence((args[arg_idx],), arg_def, def_idx)
            arg_idx += 1


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.OPERAND],
    args: Sequence[SSAValue | Sequence[SSAValue] | None],
    arg_defs: Sequence[tuple[str, OperandDef]],
    error_prefix: str,
) -> tuple[list[SSAValue], list[int]]: ...


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.RESULT],
    args: Sequence[Attribute | Sequence[Attribute] | None],
    arg_defs: Sequence[tuple[str, ResultDef]],
    error_prefix: str,
) -> tuple[list[Attribute], list[int]]: ...


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.REGION],
    args: Sequence[Region | Sequence[Region] | None],
    arg_defs: Sequence[tuple[str, RegionDef]],
    error_prefix: str,
) -> tuple[list[Region], list[int]]: ...


@overload
def irdl_build_arg_list(
    construct: Literal[VarIRConstruct.SUCCESSOR],
    args: Sequence[Successor | Sequence[Successor] | None],
    arg_defs: Sequence[tuple[str, SuccessorDef]],
    error_prefix: str,
) -> tuple[list[Successor], list[int]]: ...


_T = TypeVar("_T")


def irdl_build_arg_list(
    construct: VarIRConstruct,
    args: Sequence[_T | Sequence[_T] | None],
    arg_defs: Sequence[tuple[str, Any]],
    error_prefix: str = "",
) -> tuple[list[_T], list[int]]:
    """Build a list of arguments (operands, results, regions)"""

    if len(args) != len(arg_defs):
        raise ValueError(
            f"Expected {len(arg_defs)} {get_construct_name(construct)}, "
            f"but got {len(args)}"
        )

    res = list[_T]()
    arg_sizes = list[int]()

    for arg_idx, ((arg_name, arg_def), arg) in enumerate(zip(arg_defs, args)):
        if arg is None:
            if not isinstance(arg_def, OptionalDef):
                raise ValueError(
                    error_prefix
                    + f"passed None to a non-optional {construct} {arg_idx} '{arg_name}'"
                )
            arg_sizes.append(0)
        elif isinstance(arg, Sequence):
            if not isinstance(arg_def, VariadicDef):
                raise ValueError(
                    error_prefix
                    + f"passed Sequence to non-variadic {construct} {arg_idx} '{arg_name}'"
                )
            arg = cast(Sequence[_T], arg)

            # Check we have at most one argument for optional defintions.
            if isinstance(arg_def, OptionalDef) and len(arg) > 1:
                raise ValueError(
                    error_prefix + f"optional {construct} {arg_idx} '{arg_name}' "
                    "expects a list of size at most 1 or None, but "
                    f"got a list of size {len(arg)}"
                )

            res.extend(arg)
            arg_sizes.append(len(arg))
        else:
            res.append(arg)
            arg_sizes.append(1)
    return res, arg_sizes


_OperandArg: TypeAlias = SSAValue | Operation


def irdl_build_operations_arg(
    operand: _OperandArg | Sequence[_OperandArg] | None,
) -> SSAValue | list[SSAValue]:
    if operand is None:
        return []
    elif isinstance(operand, SSAValue):
        return operand
    elif isinstance(operand, Operation):
        return SSAValue.get(operand)
    else:
        return [SSAValue.get(op) for op in operand]


_RegionArg: TypeAlias = Region | Sequence[Operation] | Sequence[Block]


def irdl_build_region_arg(r: _RegionArg) -> Region:
    if isinstance(r, Region):
        return r

    if not len(r):
        return Region()

    if isinstance(r[0], Operation):
        ops = cast(Sequence[Operation], r)
        return Region(Block(ops))
    else:
        return Region(cast(Sequence[Block], r))


def irdl_build_regions_arg(
    r: _RegionArg | Sequence[_RegionArg] | None,
) -> Region | list[Region]:
    if r is None:
        return []
    elif isinstance(r, Region):
        return r
    elif not len(r):
        return []
    elif isinstance(r[0], Operation):
        ops = cast(Sequence[Operation], r)
        return Region(Block(ops))
    elif isinstance(r[0], Block):
        blocks = cast(Sequence[Block], r)
        return Region(blocks)
    else:
        return [irdl_build_region_arg(_r) for _r in cast(Sequence[_RegionArg], r)]


def irdl_op_init(
    self: IRDLOperation,
    op_def: OpDef,
    *,
    operands: Sequence[SSAValue | Operation | Sequence[SSAValue | Operation] | None],
    result_types: Sequence[Attribute | Sequence[Attribute] | None],
    properties: Mapping[str, Attribute | None],
    attributes: Mapping[str, Attribute | None],
    successors: Sequence[Successor | Sequence[Successor] | None],
    regions: Sequence[
        Region
        | Sequence[Operation]
        | Sequence[Block]
        | Sequence[Region | Sequence[Operation] | Sequence[Block]]
        | None
    ],
):
    """Builder for an irdl operation."""

    # We need irdl to define DenseArrayBase, but here we need
    # DenseArrayBase.
    # So we have a circular dependency that we solve by importing in this function.
    from xdsl.dialects.builtin import DenseArrayBase, i32

    error_prefix = f"Error in {op_def.name} builder: "

    operands_arg = [irdl_build_operations_arg(operand) for operand in operands]

    regions_arg = [irdl_build_regions_arg(region) for region in regions]

    # Build the operands
    built_operands, operand_sizes = irdl_build_arg_list(
        VarIRConstruct.OPERAND, operands_arg, op_def.operands, error_prefix
    )

    # Build the results
    built_res_types, result_sizes = irdl_build_arg_list(
        VarIRConstruct.RESULT, result_types, op_def.results, error_prefix
    )

    # Build the regions
    built_regions, region_sizes = irdl_build_arg_list(
        VarIRConstruct.REGION, regions_arg, op_def.regions, error_prefix
    )

    # Build the successors
    built_successors, successor_sizes = irdl_build_arg_list(
        VarIRConstruct.SUCCESSOR, successors, op_def.successors, error_prefix
    )

    # Remove all None properties
    built_properties = dict[str, Attribute]()
    for attr_name, attr in properties.items():
        if attr is None:
            continue
        built_properties[attr_name] = attr

    # Remove all None attributes
    built_attributes = dict[str, Attribute]()
    for attr_name, attr in attributes.items():
        if attr is None:
            continue
        built_attributes[attr_name] = attr

    # Take care of variadic operand and result segment sizes.
    for option in op_def.options:
        match option:
            case AttrSizedSegments():
                container = built_properties if option.as_property else built_attributes
                match option:
                    case AttrSizedOperandSegments():
                        container[AttrSizedOperandSegments.attribute_name] = (
                            DenseArrayBase.from_list(i32, operand_sizes)
                        )

                    case AttrSizedResultSegments():
                        container[AttrSizedResultSegments.attribute_name] = (
                            DenseArrayBase.from_list(i32, result_sizes)
                        )

                    case AttrSizedRegionSegments():
                        container[AttrSizedRegionSegments.attribute_name] = (
                            DenseArrayBase.from_list(i32, region_sizes)
                        )

                    case AttrSizedSuccessorSegments():
                        container[AttrSizedSuccessorSegments.attribute_name] = (
                            DenseArrayBase.from_list(i32, successor_sizes)
                        )
                    case _:
                        raise ValueError(
                            f"Unexpected option {option} in operation definition {op_def}."
                        )
            case SameVariadicSize():
                match option:
                    case SameVariadicOperandSize():
                        sizes = operand_sizes
                        construct = VarIRConstruct.OPERAND
                    case SameVariadicResultSize():
                        sizes = result_sizes
                        construct = VarIRConstruct.RESULT
                    case SameVariadicRegionSize():
                        sizes = region_sizes
                        construct = VarIRConstruct.REGION
                    case SameVariadicSuccessorSize():
                        sizes = successor_sizes
                        construct = VarIRConstruct.SUCCESSOR
                    case _:
                        raise ValueError(
                            f"Unexpected option {option} in operation definition {op_def}."
                        )
                variadic_sizes = [
                    size
                    for (size, def_) in zip(
                        sizes, get_construct_defs(op_def, construct)
                    )
                    if isinstance(def_[1], VariadicDef)
                ]
                if any(size != variadic_sizes[0] for size in variadic_sizes[1:]):
                    raise ValueError(
                        f"Variadic {get_construct_name(construct)}s have different sizes: {variadic_sizes}"
                    )
            case _:
                pass

    Operation.__init__(
        self,
        operands=built_operands,
        result_types=built_res_types,
        properties=built_properties,
        attributes=built_attributes,
        successors=built_successors,
        regions=built_regions,
    )


def irdl_op_arg_definition(
    new_attrs: dict[str, Any], construct: VarIRConstruct, op_def: OpDef
) -> None:
    previous_variadics = 0
    defs = get_construct_defs(op_def, construct)
    for arg_idx, (arg_name, arg_def) in enumerate(defs):

        def fun(self: Any, idx: int = arg_idx, previous_vars: int = previous_variadics):
            return get_operand_result_or_region(
                self, op_def, idx, previous_vars, construct
            )

        new_attrs[arg_name] = property(fun)
        if isinstance(arg_def, VariadicDef):
            previous_variadics += 1

    # If we have multiple variadics, check that we have an
    # attribute that holds the variadic sizes.
    variadics_option = get_multiple_variadic_options(construct)
    if previous_variadics > 1 and (
        not any(
            isinstance(o, option) for o in op_def.options for option in variadics_option
        )
    ):
        names = list(option.__name__ for option in variadics_option)
        names, last_name = names[:-1], names[-1]
        raise PyRDLOpDefinitionError(
            f"Operation {op_def.name} defines more than two variadic "
            f"{get_construct_name(construct)}s, but do not define any of "
            f"{', '.join(names)} or {last_name} PyRDL options."
        )


def _optional_attribute_field(attribute_name: str, default_value: Attribute | None):
    """Returns the getter and setter for an optional operation attribute."""

    def field_getter(self: IRDLOperation):
        return self.attributes.get(attribute_name, default_value)

    def field_setter(self: IRDLOperation, value: Attribute | None):
        if value is None:
            self.attributes.pop(attribute_name, None)
        else:
            self.attributes[attribute_name] = value

    return property(field_getter, field_setter)


def _attribute_field(attribute_name: str):
    """Returns the getter and setter for an operation attribute."""

    def field_getter(self: IRDLOperation):
        return self.attributes[attribute_name]

    def field_setter(self: IRDLOperation, value: Attribute):
        self.attributes[attribute_name] = value

    return property(field_getter, field_setter)


def _optional_property_field(property_name: str, default_value: Attribute | None):
    """Returns the getter and setter for an optional operation property."""

    def field_getter(self: IRDLOperation):
        return self.properties.get(property_name, default_value)

    def field_setter(self: IRDLOperation, value: Attribute | None):
        if value is None:
            self.properties.pop(property_name, None)
        else:
            self.properties[property_name] = value

    return property(field_getter, field_setter)


def _property_field(property_name: str):
    """Returns the getter and setter for an operation property."""

    def field_getter(self: IRDLOperation):
        return self.properties[property_name]

    def field_setter(self: IRDLOperation, value: Attribute):
        self.properties[property_name] = value

    return property(field_getter, field_setter)


def get_accessors_from_op_def(
    op_def: OpDef, custom_verify: Any | None
) -> dict[str, Any]:
    """Get python accessors from an operation definition."""
    new_attrs = dict[str, Any]()

    # Add operand access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.OPERAND, op_def)

    # Add result access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.RESULT, op_def)

    # Add region access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.REGION, op_def)

    # Add successor access fields
    irdl_op_arg_definition(new_attrs, VarIRConstruct.SUCCESSOR, op_def)

    for accessor_name, (
        attribute_name,
        attribute_type,
    ) in op_def.accessor_names.items():
        if attribute_type == "attribute":
            attr_def = op_def.attributes[attribute_name]
            if isinstance(attr_def, OptAttributeDef):
                new_attrs[accessor_name] = _optional_attribute_field(
                    attribute_name, op_def.attributes[attribute_name].default_value
                )
            else:
                new_attrs[accessor_name] = _attribute_field(attribute_name)
        else:
            prop_def = op_def.properties[attribute_name]
            if isinstance(prop_def, OptPropertyDef):
                new_attrs[accessor_name] = _optional_property_field(
                    attribute_name, op_def.properties[attribute_name].default_value
                )
            else:
                new_attrs[accessor_name] = _property_field(attribute_name)

    # If the traits are already defined then this is a no-op, as the new attrs are
    # passed after the existing attrs, otherwise this sets an empty OpTraits.
    new_attrs["traits"] = op_def.traits

    @classmethod
    def get_irdl_definition(cls: type[IRDLOperationInvT]):
        return op_def

    new_attrs["get_irdl_definition"] = get_irdl_definition

    if op_def.assembly_format is not None:
        from xdsl.irdl.declarative_assembly_format import FormatProgram

        try:
            assembly_program = FormatProgram.from_str(op_def.assembly_format, op_def)
        except ParseError as e:
            raise PyRDLOpDefinitionError(
                "Error during the parsing of the assembly format: ", e.args
            ) from e

        @classmethod
        def parse_with_format(
            cls: type[IRDLOperationInvT], parser: Parser
        ) -> IRDLOperationInvT:
            return assembly_program.parse(parser, cls)

        def print_with_format(self: IRDLOperation, printer: Printer):
            return assembly_program.print(printer, self)

        new_attrs["parse"] = parse_with_format
        new_attrs["print"] = print_with_format

    if custom_verify is not None:

        def verify_(self: IRDLOperation):
            op_def.verify(self)
            custom_verify(self)

        new_attrs["verify_"] = verify_
    else:

        def verify_(self: IRDLOperation):
            op_def.verify(self)

        new_attrs["verify_"] = verify_

    return new_attrs


def irdl_op_definition(cls: type[IRDLOperationInvT]) -> type[IRDLOperationInvT]:
    """Decorator used on classes to define a new operation definition."""

    assert issubclass(cls, IRDLOperation), (
        f"class {cls.__name__} should be a subclass of IRDLOperation"
    )

    op_def = OpDef.from_pyrdl(cls)
    new_attrs = get_accessors_from_op_def(op_def, getattr(cls, "verify_", None))

    return type.__new__(
        type(cls), cls.__name__, cls.__mro__, {**cls.__dict__, **new_attrs}
    )
