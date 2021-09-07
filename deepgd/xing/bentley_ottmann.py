from functools import partial
from itertools import product, repeat, combinations
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Union,
    TypeVar,
    Callable,
    Hashable,
    Iterable,
    Optional,
    Sequence
)

from dendroid import red_black
from prioq.base import PriorityQueue
from ground.base import (
    get_context,
    Relation,
    Context,
    Orientation
)
from ground.hints import (
    Point,
    Segment
)


T = TypeVar('T')
Event = TypeVar('Event')
Intersection = Union[Tuple[Point], Tuple[Point, Point]]


def classify_overlap(test_start: Point,
                     test_end: Point,
                     goal_start: Point,
                     goal_end: Point) -> Relation:
    assert test_start < test_end
    assert goal_start < goal_end
    if test_start == goal_start:
        return (Relation.COMPONENT
                if test_end < goal_end
                else (Relation.COMPOSITE
                      if goal_end < test_end
                      else Relation.EQUAL))
    elif test_end == goal_end:
        return (Relation.COMPOSITE
                if test_start < goal_start
                else Relation.COMPONENT)
    elif goal_start < test_start < goal_end:
        return (Relation.COMPONENT
                if test_end < goal_end
                else Relation.OVERLAP)
    else:
        assert test_start < goal_start < test_end
        return (Relation.COMPOSITE
                if goal_end < test_end
                else Relation.OVERLAP)


class LeftEvent:
    @classmethod
    def fromSegment(cls, segment: Segment, segment_id: int) -> 'LeftEvent':
        start, end = sorted((segment.start, segment.end))
        result = LeftEvent(start, None, start, {start: {end: {segment_id}}})
        result.right = RightEvent(end, result, end)
        return result

    __slots__ = ()

    is_left = True

    @property
    def end(self) -> Point:
        return self.right.start

    @property
    def original_start(self) -> Point:
        return self._original_start

    @property
    def original_end(self) -> Point:
        return self.right.original_start

    @property
    def segments_ids(self) -> Set[int]:
        return self.parts_ids[self.start][self.end]

    @property
    def start(self) -> Point:
        return self._start

    @property
    def tangents(self) -> Sequence[Event]:
        return self.Tangents

    __slots__ = ('parts_ids', 'right', '_original_start', '_relations_mask',
                 '_start', 'Tangents')

    def __init__(self,
                 start: Point,
                 right: Optional['RightEvent'],
                 original_start: Point,
                 parts_ids: Dict[Point, Dict[Point, Set[int]]]) -> None:
        self.right, self.parts_ids, self._original_start, self._start = (
            right, parts_ids, original_start, start)
        self._relations_mask = 0
        self.Tangents = []  # type: List[Event]

    def divide(self, break_point: Point) -> 'LeftEvent':
        """Divides the event at given break point and returns tail."""
        segments_ids = self.segments_ids
        (self.parts_ids.setdefault(self.start, {})
         .setdefault(break_point, set()).update(segments_ids))
        (self.parts_ids.setdefault(break_point, {})
         .setdefault(self.end, set()).update(segments_ids))
        result = self.right.left = LeftEvent(
                break_point, self.right, self.original_start,
                self.parts_ids)
        self.right = RightEvent(break_point, self, self.original_end)
        return result

    def has_only_relations(self, *relations: Relation) -> bool:
        mask = self._relations_mask
        for relation in relations:
            mask &= ~(1 << relation)
        return not mask

    def merge_with(self, other: 'LeftEvent') -> None:
        assert self.start == other.start and self.end == other.end
        full_relation = classify_overlap(
                other.original_start, other.original_end, self.original_start,
                self.original_end)
        self.register_relation(full_relation)
        other.register_relation(full_relation.complement)
        start, end = self.start, self.end
        self.parts_ids[start][end] = other.parts_ids[start][end] = (
                self.parts_ids[start][end] | other.parts_ids[start][end])

    def registerTangent(self, tangent: Event) -> None:
        assert self.start == tangent.start
        self.Tangents.append(tangent)

    def register_relation(self, relation: Relation) -> None:
        self._relations_mask |= 1 << relation


class RightEvent:
    __slots__ = ()

    is_left = False
    
    @property
    def end(self) -> Point:
        return self.left.start

    @property
    def original_end(self) -> Point:
        return self.left.original_start

    @property
    def original_start(self) -> Point:
        return self._original_start

    @property
    def segments_ids(self) -> Set[int]:
        return self.left.segments_ids

    @property
    def start(self) -> Point:
        return self._start

    @property
    def tangents(self) -> Sequence[Event]:
        return self.Tangents

    __slots__ = 'left', '_original_start', '_start', 'Tangents'

    def __init__(self,
                 start: Point,
                 left: Optional[LeftEvent],
                 original_start: Point) -> None:
        self.left, self._original_start, self._start = (left, original_start,
                                                        start)
        self.Tangents = []  # type: List[Event]

    def registerTangent(self, tangent: 'Event') -> None:
        assert self.start == tangent.start
        self.Tangents.append(tangent)


class SweepLine:
    __slots__ = 'context', '_set'

    def __init__(self, context: Context) -> None:
        self.context = context
        self._set = red_black.set_(key=partial(SweepLineKey,
                                               context.angle_orientation))

    def add(self, event: LeftEvent) -> None:
        self._set.add(event)

    def find_equal(self, event: LeftEvent) -> Optional[LeftEvent]:
        try:
            candidate = self._set.floor(event)
        except ValueError:
            return None
        else:
            return (candidate
                    if (candidate.start == event.start
                        and candidate.end == event.end)
                    else None)

    def remove(self, event: LeftEvent) -> None:
        self._set.remove(event)

    def above(self, event: LeftEvent) -> Optional[LeftEvent]:
        try:
            return self._set.next(event)
        except ValueError:
            return None

    def below(self, event: LeftEvent) -> Optional[LeftEvent]:
        try:
            return self._set.prev(event)
        except ValueError:
            return None


class SweepLineKey:
    __slots__ = 'event', 'orienteer'

    def __init__(self,
                 orienteer: Callable[[Point, Point, Point], Orientation],
                 event: LeftEvent) -> None:
        self.event, self.orienteer = event, orienteer

    def __lt__(self, other: 'SweepLineKey') -> bool:
        """
        Checks if the segment (or at least the point) associated with event
        is lower than other's.
        """
        event, other_event = self.event, other.event
        if event is other_event:
            return False
        start, other_start = event.start, other_event.start
        end, other_end = event.end, other_event.end
        other_start_orientation = self.orienteer(start, end, other_start)
        other_end_orientation = self.orienteer(start, end, other_end)
        if other_start_orientation is other_end_orientation:
            start_x, start_y = start.x, start.y
            other_start_x, other_start_y = other_start.x, other_start.y
            if other_start_orientation is not Orientation.COLLINEAR:
                # other segment fully lies on one side
                return other_start_orientation is Orientation.COUNTERCLOCKWISE
            # segments are collinear
            elif start_x == other_start_x:
                end_x, end_y = end.x, end.y
                other_end_x, other_end_y = other_end.x, other_end.y
                if start_y != other_start_y:
                    # segments are vertical
                    return start_y < other_start_y
                # segments have same start
                elif end_y != other_end_y:
                    return end_y < other_end_y
                else:
                    return end_x < other_end_x
            elif start_y != other_start_y:
                return start_y < other_start_y
            else:
                # segments are horizontal
                return start_x < other_start_x
        start_orientation = self.orienteer(other_start, other_end, start)
        end_orientation = self.orienteer(other_start, other_end, end)
        if start_orientation is end_orientation:
            return start_orientation is Orientation.CLOCKWISE
        elif other_start_orientation is Orientation.COLLINEAR:
            return other_end_orientation is Orientation.COUNTERCLOCKWISE
        elif start_orientation is Orientation.COLLINEAR:
            return end_orientation is Orientation.CLOCKWISE
        elif end_orientation is Orientation.COLLINEAR:
            return start_orientation is Orientation.CLOCKWISE
        else:
            return other_start_orientation is Orientation.COUNTERCLOCKWISE


class EventsQueue:
    @classmethod
    def fromSegments(cls,
                      segments: Sequence[Segment],
                      *,
                      context: Context) -> 'EventsQueue':
        result = cls(context)
        for index, segment in enumerate(segments):
            event = LeftEvent.fromSegment(segment, index)
            result.push(event)
            result.push(event.right)
        return result

    __slots__ = 'context', '_queue'

    def __init__(self, context: Context) -> None:
        self.context = context
        self._queue = PriorityQueue(key=EventsQueueKey)

    def __bool__(self) -> bool:
        return bool(self._queue)

    def detectIntersection(self,
                            below_event: LeftEvent,
                            event: LeftEvent,
                            sweep_line: SweepLine) -> None:
        relation = self.context.segments_relation(below_event, event)
        if relation is Relation.DISJOINT:
            return
        elif relation is Relation.TOUCH or relation is Relation.CROSS:
            # segments touch or cross
            point = self.context.segments_intersection(below_event, event)
            assert event.segments_ids.isdisjoint(below_event.segments_ids)
            if point != below_event.start and point != below_event.end:
                below_below = sweep_line.below(below_event)
                assert not (below_below is not None
                            and below_below.start == below_event.start
                            and below_below.end == point)
                self.push(below_event.divide(point))
                self.push(below_event.right)
            if point != event.start and point != event.end:
                above_event = sweep_line.above(event)
                if (above_event is not None
                        and above_event.start == event.start
                        and above_event.end == point):
                    sweep_line.remove(above_event)
                    self.push(event.divide(point))
                    self.push(event.right)
                    event.merge_with(above_event)
                else:
                    self.push(event.divide(point))
                    self.push(event.right)
        else:
            # segments overlap
            starts_equal = event.start == below_event.start
            start_min, start_max = (
                (event, below_event)
                if (starts_equal
                    or EventsQueueKey(event) < EventsQueueKey(below_event))
                else (below_event, event))
            ends_equal = event.end == below_event.end
            end_min, end_max = (
                (event.right, below_event.right)
                if ends_equal or (EventsQueueKey(event.right)
                                  < EventsQueueKey(below_event.right))
                else (below_event.right, event.right))
            if starts_equal:
                assert not ends_equal
                # segments share the left endpoint
                sweep_line.remove(end_max.left)
                self.push(end_max.left.divide(end_min.start))
                event.merge_with(below_event)
            elif ends_equal:
                # segments share the right endpoint
                start_max.merge_with(start_min.divide(start_max.start))
                self.push(start_min.right)
            elif start_min is end_max.left:
                # one line segment includes the other one
                self.push(start_min.divide(end_min.start))
                self.push(start_min.right)
                start_max.merge_with(start_min.divide(start_max.start))
                self.push(start_min.right)
            else:
                # no line segment includes the other one
                self.push(start_max.divide(end_min.start))
                start_max.merge_with(start_min.divide(start_max.start))
                self.push(start_max.right)
                self.push(start_min.right)

    def peek(self) -> Event:
        return self._queue.peek()

    def pop(self) -> Event:
        return self._queue.pop()

    def push(self, event: Event) -> None:
        if event.start == event.end:
            raise ValueError('Degenerate segment found '
                             'with both endpoints being: {}.'
                             .format(event.start))
        self._queue.push(event)


class EventsQueueKey:
    __slots__ = 'event',

    def __init__(self, event: Event) -> None:
        self.event = event

    def __lt__(self, other: 'EventsQueueKey') -> bool:
        """
        Checks if the event should be processed before the other.
        """
        event, other_event = self.event, other.event
        start_x, start_y = event.start.x, event.start.y
        other_start_x, other_start_y = other_event.start.x, other_event.start.y
        if start_x != other_start_x:
            # different x-coordinate,
            # the event with lower x-coordinate is processed first
            return start_x < other_start_x
        elif start_y != other_start_y:
            # different starts, but same x-coordinate,
            # the event with lower y-coordinate is processed first
            return start_y < other_start_y
        elif event.is_left is not other_event.is_left:
            # same start, but one is a left endpoint
            # and the other is a right endpoint,
            # the right endpoint is processed first
            return not event.is_left
        else:
            # same start,
            # both events are left endpoints or both are right endpoints
            return event.end < other_event.end


def sweep(segments: Sequence[Segment],
          *,
          context: Context) -> Iterable[LeftEvent]:
    events_queue = EventsQueue.fromSegments(segments,
                                             context=context)
    sweep_line = SweepLine(context)
    start = (events_queue.peek().start
             if events_queue
             else None)  # type: Optional[Point]
    same_start_events = []  # type: List[Event]
    while events_queue:
        event = events_queue.pop()
        if event.start == start:
            same_start_events.append(event)
        else:
            yield from complete_events_relations(same_start_events)
            same_start_events, start = [event], event.start
        if event.is_left:
            equalSegment_event = sweep_line.find_equal(event)
            if equalSegment_event is None:
                sweep_line.add(event)
                below_event = sweep_line.below(event)
                if below_event is not None:
                    events_queue.detectIntersection(below_event, event,
                                                     sweep_line)
                above_event = sweep_line.above(event)
                if above_event is not None:
                    events_queue.detectIntersection(event, above_event,
                                                     sweep_line)
            else:
                # found equal segments' fragments
                equalSegment_event.merge_with(event)
        else:
            event = event.left
            equalSegment_event = sweep_line.find_equal(event)
            if equalSegment_event is not None:
                above_event, below_event = (
                    sweep_line.above(equalSegment_event),
                    sweep_line.below(equalSegment_event))
                sweep_line.remove(equalSegment_event)
                if below_event is not None and above_event is not None:
                    events_queue.detectIntersection(below_event, above_event,
                                                     sweep_line)
                if event is not equalSegment_event:
                    equalSegment_event.merge_with(event)
    yield from complete_events_relations(same_start_events)


def complete_events_relations(same_start_events: Sequence[Event]
                              ) -> Iterable[Event]:
    for offset, first in enumerate(same_start_events,
                                   start=1):
        first_left = first if first.is_left else first.left
        first_ids = first_left.segments_ids
        for second_index in range(offset, len(same_start_events)):
            second = same_start_events[second_index]
            second_left = second if second.is_left else second.left
            second_ids = second_left.segments_ids
            first_extra_ids_count, second_extra_ids_count = (
                len(first_ids - second_ids), len(second_ids - first_ids))
            if first_extra_ids_count and second_extra_ids_count:
                relation = (Relation.TOUCH
                            if (first.start == first.original_start
                                or second.start == second.original_start)
                            else Relation.CROSS)
                first.registerTangent(second)
                second.registerTangent(first)
                first_left.register_relation(relation)
                second_left.register_relation(relation.complement)
            elif first_extra_ids_count or second_extra_ids_count:
                relation = classify_overlap(first_left.original_start,
                                            first_left.original_end,
                                            second_left.original_start,
                                            second_left.original_end)
                first_left.register_relation(relation)
                second_left.register_relation(relation.complement)
        if not first.is_left:
            yield first_left


def segments_intersections(segments: Sequence[Segment],
                           *,
                           context: Optional[Context] = None
                           ) -> Dict[Tuple[int, int], Intersection]:
    left_parts_ids, right_parts_ids = {}, {}
    leftTangents, rightTangents = {}, {}
    for event in sweep(
            segments,
            context=get_context() if context is None else context):
        if event.tangents:
            (leftTangents.setdefault(event.start, {})
             .setdefault(event.end, set())
             .update(tangent.end for tangent in event.tangents))
        if event.right.tangents:
            (rightTangents.setdefault(event.end, {})
             .setdefault(event.start, set())
             .update(tangent.end for tangent in event.right.tangents))
        for start, ends_ids in event.parts_ids.items():
            for end, ids in ends_ids.items():
                (left_parts_ids.setdefault(start, {}).setdefault(end, set())
                 .update(ids))
                (right_parts_ids.setdefault(end, {}).setdefault(start, set())
                 .update(ids))
    discrete = {}  # type: Dict[Tuple[int, int], Tuple[Point]]
    for intersection_point, endsTangents_ends in leftTangents.items():
        leftIntersection_point_ids, rightIntersection_point_ids = (
            left_parts_ids.get(intersection_point),
            right_parts_ids.get(intersection_point))
        for end, tangents_ends in endsTangents_ends.items():
            ids = leftIntersection_point_ids[end]
            for tangent_end in tangents_ends:
                tangent_ids = (leftIntersection_point_ids[tangent_end]
                               if intersection_point < tangent_end
                               else rightIntersection_point_ids[tangent_end])
                ids_pairs = [
                    tuple(sorted((id_, tangent_id)))
                    for id_, tangent_id in product(ids - tangent_ids,
                                                    tangent_ids - ids)]
                discrete.update(zip(ids_pairs, repeat((intersection_point,))))
    for intersection_point, startsTangents_ends in rightTangents.items():
        leftIntersection_point_ids, rightIntersection_point_ids = (
            left_parts_ids.get(intersection_point),
            right_parts_ids.get(intersection_point))
        for start, tangents_ends in startsTangents_ends.items():
            ids = rightIntersection_point_ids[start]
            for tangent_end in tangents_ends:
                tangent_ids = (leftIntersection_point_ids[tangent_end]
                               if intersection_point < tangent_end
                               else rightIntersection_point_ids[tangent_end])
                ids_pairs = [
                    tuple(sorted((id_, tangent_id)))
                    for id_, tangent_id in product(ids - tangent_ids,
                                                    tangent_ids - ids)]
                discrete.update(zip(ids_pairs, repeat((intersection_point,))))
    continuous = {}  # type: Dict[Tuple[int, int], Tuple[Point, Point]]
    for start, ends_ids in left_parts_ids.items():
        for end, ids in ends_ids.items():
            for ids_pair in combinations(sorted(ids), r=2):
                if ids_pair in continuous:
                    prev_start, prev_end = continuous[ids_pair]
                    endpoints = min(prev_start, start), max(prev_end, end)
                else:
                    endpoints = (start, end)
                continuous[ids_pair] = endpoints
    return {**discrete, **continuous}
