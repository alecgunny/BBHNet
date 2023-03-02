import numpy as np

from bbhnet.analysis.ledger import events


class TestTimeSlideEventSet:
    def test_append(self):
        det_stats = np.random.randn(10)
        times = np.arange(10)
        obj1 = events.TimeSlideEventSet(det_stats, times, 100)
        obj2 = events.TimeSlideEventSet(-det_stats, times + 10, 50)
        obj1.append(obj2)
        assert obj1.Tb == 150
        det_stats = np.split(obj1.detection_statistic, 2)
        assert (det_stats[0] == -det_stats[1]).all()
        times = np.split(obj1.time, 2)
        assert (times[0] == times[1] - 10).all()

    def test_nb(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        obj = events.TimeSlideEventSet(det_stats, times, 100)
        assert obj.nb(5) == 5
        assert obj.nb(5.5) == 4
        assert obj.nb(-1) == 10

        assert (obj.nb(np.array([5, 5.5])) == np.array([5, 4])).all()

    def test_far(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        Tb = 2 * events.SECONDS_IN_YEAR
        obj = events.TimeSlideEventSet(det_stats, times, Tb)

        assert obj.far(5) == 2.5
        assert (obj.far(np.array([5, 5.5])) == np.array([2.5, 2])).all()

    # TODO: add a test for significance that doen'st
    # just replicate the logic of the function itself


class TestEventSet:
    def test_from_timeslide(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        obj = events.TimeSlideEventSet(det_stats, times, 100)

        shift = [0]
        supobj = events.EventSet.from_timeslide(obj, shift)
        assert len(supobj) == 10
        assert supobj.Tb == 100
        assert supobj.shift.ndim == 2
        assert (supobj.shift == 0).all()

        shift = [0, 1]
        supobj = events.EventSet.from_timeslide(obj, shift)
        assert len(supobj) == 10
        assert supobj.Tb == 100
        assert supobj.shift.ndim == 2
        assert (supobj.shift == np.array([0, 1])).all()

    def test_get_shift(self):
        det_stats = np.arange(10)
        times = np.arange(10)
        shifts = np.array([0] * 4 + [1] * 5 + [2])
        obj = events.EventSet(det_stats, times, 100, shifts)

        subobj = obj.get_shift(0)
        assert len(subobj) == 4
        assert (subobj.detection_statistic == np.arange(4)).all()
        assert (subobj.shift == 0).all()

        subobj = obj.get_shift(2)
        assert len(subobj) == 1
        assert (subobj.detection_statistic == 9).all()
        assert (subobj.shift == 2).all()

        shifts = np.array([[0, 0]] * 5 + [[0, 1]] * 2 + [[1, 1]] * 3)
        obj = events.EventSet(det_stats, times, 100, shifts)
        subobj = obj.get_shift(np.array([0, 1]))
        assert len(subobj) == 2
        assert (subobj.detection_statistic == np.arange(5, 7)).all()
        assert (subobj.shift == np.array([0, 1])).all()
