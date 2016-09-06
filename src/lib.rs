// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! A cache that holds a limited size of key-value pairs.
//! When the capacity of the cache is exceeded, the least-recently-used
//! pairs are automatically removed.
//! (where "used" means a look-up or putting the pair into the cache)
//!
//! It is a logic error for an item to be modified in such a way that changes the size.
//! This is normally only possible through Cell, RefCell, global state, I/O, or unsafe code.
//!
//! # Examples
//!
//! ```
//! use lru_size_cache::LruSizeCache;
//!
//! let mut cache = LruSizeCache::new(2);
//!
//! cache.insert(1, "1");
//! cache.insert(2, "2");
//! cache.insert(3, "3");
//! assert!(cache.get(&1).is_none());
//! assert_eq!(*cache.get(&2).unwrap(), "2");
//! assert_eq!(*cache.get(&3).unwrap(), "3");
//!
//! cache.insert(2, "4");
//! assert_eq!(*cache.get(&2).unwrap(), "4");
//!
//! cache.insert(6, "6");
//! assert!(cache.get(&3).is_none());
//!
//! cache.set_capacity(1);
//! assert!(cache.get(&2).is_none());
//! ```

extern crate linked_hash_map;

use std::borrow::Borrow;
use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{Hash, BuildHasher};

use linked_hash_map::LinkedHashMap;

pub trait HasSize {
    fn size(&self) -> usize;
}

impl<'a> HasSize for &'a str {
    fn size(&self) -> usize { self.len() }
}

// TODO maybe implement `HasSize` for other `std` types (like collections)

/// An LRU cache.
#[derive(Clone)]
pub struct LruSizeCache<K: Eq + Hash, V, S: BuildHasher = RandomState> {
    map: LinkedHashMap<K, V, S>,
    capacity: usize,
    used: usize,
}

impl<K: Eq + Hash, V> LruSizeCache<K, V> {
    /// Creates an empty cache that can hold at most `capacity` items.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    /// let mut cache: LruSizeCache<i32, &str> = LruSizeCache::new(10);
    /// ```
    pub fn new(capacity: usize) -> Self {
        LruSizeCache {
            map: LinkedHashMap::new(),
            capacity: capacity,
            used: 0,
        }
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> LruSizeCache<K, V, S> {
    /// Creates an empty cache that can hold at most `capacity` items with the given hash builder.
    pub fn with_hasher(capacity: usize, hash_builder: S) -> Self {
        LruSizeCache {
            map: LinkedHashMap::with_hasher(hash_builder),
            capacity: capacity,
            used: 0,
        }
    }

    /// Checks if the map contains the given key.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    ///
    /// let mut cache = LruSizeCache::new(1);
    ///
    /// cache.insert(1, "a");
    /// assert_eq!(cache.contains_key(&1), true);
    /// ```
    pub fn contains_key<Q: ?Sized>(&mut self, key: &Q) -> bool
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        self.get(key).is_some()
    }

    /// Returns a reference to the value corresponding to the given key in the cache, if any.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    ///
    /// let mut cache = LruSizeCache::new(2);
    ///
    /// cache.insert(1, "a");
    /// cache.insert(2, "b");
    /// cache.insert(2, "c");
    /// cache.insert(3, "d");
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), Some(&"c"));
    /// ```
    pub fn get<Q: ?Sized>(&mut self, k: &Q) -> Option<&V>
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        self.map.get_refresh(k).map(|it| it as &_)
    }

    /// Returns the maximum number of key-value pairs the cache can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    /// let mut cache: LruSizeCache<i32, &str> = LruSizeCache::new(2);
    /// assert_eq!(cache.capacity(), 2);
    /// ```
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the used space in the cache.
    pub fn used(&self) -> usize { self.used }

    /// Returns the number of key-value pairs in the cache.
    pub fn len(&self) -> usize { self.map.len() }

    /// Returns `true` if the cache contains no key-value pairs.
    pub fn is_empty(&self) -> bool { self.map.is_empty() }

    /// Removes all key-value pairs from the cache.
    pub fn clear(&mut self) { self.map.clear(); self.used = 0; }

    /// Returns an iterator over the cache's key-value pairs in least- to most-recently-used order.
    ///
    /// Accessing the cache through the iterator does _not_ affect the cache's LRU state.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    ///
    /// let mut cache = LruSizeCache::new(2);
    ///
    /// cache.insert(1, "1");
    /// cache.insert(2, "2");
    /// cache.insert(3, "3");
    ///
    /// let kvs: Vec<_> = cache.iter().collect();
    /// assert_eq!(kvs, [(&2, &"2"), (&3, &"3")]);
    /// ```
    pub fn iter(&self) -> Iter<K, V> { Iter(self.map.iter()) }
}

impl<K: Eq + Hash, V: HasSize, S: BuildHasher> LruSizeCache<K, V, S> {
    /// Inserts a key-value pair into the cache. If the key already existed, the old value is
    /// returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    ///
    /// let mut cache = LruSizeCache::new(2);
    ///
    /// cache.insert(1, "a");
    /// cache.insert(2, "b");
    /// assert_eq!(cache.get(&1), Some(&"a"));
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// ```
    pub fn insert(&mut self, k: K, v: V) -> Option<V> {
        self.used += v.size();
        let old = self.map.insert(k, v);
        if let Some(ref old_val) = old {
            self.used -= old_val.size()
        }
        while self.used() > self.capacity() {
            self.remove_lru();
        }
        old
    }

    /// Removes the given key from the cache and returns its corresponding value.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    ///
    /// let mut cache = LruSizeCache::new(2);
    ///
    /// cache.insert(2, "a");
    ///
    /// assert_eq!(cache.remove(&1), None);
    /// assert_eq!(cache.remove(&2), Some("a"));
    /// assert_eq!(cache.remove(&2), None);
    /// assert_eq!(cache.len(), 0);
    /// ```
    pub fn remove<Q: ?Sized>(&mut self, k: &Q) -> Option<V>
        where K: Borrow<Q>,
              Q: Hash + Eq
    {
        let removed = self.map.remove(k);
        if let Some(ref removed_val) = removed {
            self.used -= removed_val.size();
        }
        removed
    }

    /// Sets the number of key-value pairs the cache can hold. Removes
    /// least-recently-used key-value pairs if necessary.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    ///
    /// let mut cache = LruSizeCache::new(2);
    ///
    /// cache.insert(1, "a");
    /// cache.insert(2, "b");
    /// cache.insert(3, "c");
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// assert_eq!(cache.get(&3), Some(&"c"));
    ///
    /// cache.set_capacity(3);
    /// cache.insert(1, "a");
    /// cache.insert(2, "b");
    ///
    /// assert_eq!(cache.get(&1), Some(&"a"));
    /// assert_eq!(cache.get(&2), Some(&"b"));
    /// assert_eq!(cache.get(&3), Some(&"c"));
    ///
    /// cache.set_capacity(1);
    ///
    /// assert_eq!(cache.get(&1), None);
    /// assert_eq!(cache.get(&2), None);
    /// assert_eq!(cache.get(&3), Some(&"c"));
    /// ```
    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        while self.used() > self.capacity() {
            self.remove_lru();
        }
    }

    /// Removes and returns the least recently used key-value pair as a tuple.
    ///
    /// # Examples
    ///
    /// ```
    /// use lru_size_cache::LruSizeCache;
    ///
    /// let mut cache = LruSizeCache::new(2);
    ///
    /// cache.insert(1, "a");
    /// cache.insert(2, "b");
    ///
    /// assert_eq!(cache.remove_lru(), Some((1, "a")));
    /// assert_eq!(cache.len(), 1);
    /// ```
    #[inline]
    pub fn remove_lru(&mut self) -> Option<(K, V)> {
        let removed = self.map.pop_front();
        if let Some((_, ref removed_val)) = removed {
            self.used -= removed_val.size();
        }
        removed
    }
}

impl<K: Eq + Hash, V: HasSize, S: BuildHasher> Extend<(K, V)> for LruSizeCache<K, V, S> {
    fn extend<I: IntoIterator<Item=(K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

impl<K: fmt::Debug + Eq + Hash, V: fmt::Debug, S: BuildHasher> fmt::Debug for LruSizeCache<K, V, S> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_map().entries(self.iter().rev()).finish()
    }
}

impl<K: Eq + Hash, V, S: BuildHasher> IntoIterator for LruSizeCache<K, V, S> {
    type Item = (K, V);
    type IntoIter = IntoIter<K, V>;

    fn into_iter(self) -> IntoIter<K, V> {
        IntoIter(self.map.into_iter())
    }
}

impl<'a, K: Eq + Hash, V, S: BuildHasher> IntoIterator for &'a LruSizeCache<K, V, S> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;
    fn into_iter(self) -> Iter<'a, K, V> { self.iter() }
}

/// An iterator over a cache's key-value pairs in least- to most-recently-used order.
///
/// # Examples
///
/// ```
/// use lru_size_cache::LruSizeCache;
///
/// let mut cache = LruSizeCache::new(2);
///
/// cache.insert(1, "1");
/// cache.insert(2, "2");
/// cache.insert(3, "3");
///
/// let mut n = 2;
///
/// for (k, v) in cache {
///     assert_eq!(k, n);
///     assert_eq!(v, &format!("{}", n));
///     n += 1;
/// }
///
/// assert_eq!(n, 4);
/// ```
#[derive(Clone)]
pub struct IntoIter<K, V>(linked_hash_map::IntoIter<K, V>);

impl<K, V> Iterator for IntoIter<K, V> {
    type Item = (K, V);

    fn next(&mut self) -> Option<(K, V)> {
        self.0.next()
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<K, V> DoubleEndedIterator for IntoIter<K, V> {
    fn next_back(&mut self) -> Option<(K, V)> {
        self.0.next_back()
    }
}

impl<K, V> ExactSizeIterator for IntoIter<K, V> {
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// An iterator over a cache's key-value pairs in least- to most-recently-used order.
///
/// Accessing a cache through the iterator does _not_ affect the cache's LRU state.
pub struct Iter<'a, K: 'a, V: 'a>(linked_hash_map::Iter<'a, K, V>);

impl<'a, K, V> Clone for Iter<'a, K, V> {
    fn clone(&self) -> Iter<'a, K, V> { Iter(self.0.clone()) }
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);
    fn next(&mut self) -> Option<(&'a K, &'a V)> { self.0.next() }
    fn size_hint(&self) -> (usize, Option<usize>) { self.0.size_hint() }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<(&'a K, &'a V)> { self.0.next_back() }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {
    fn len(&self) -> usize { self.0.len() }
}

#[cfg(test)]
mod tests {
    use super::LruSizeCache;

    #[test]
    fn test_put_and_get() {
        let mut cache = LruSizeCache::new(2);
        cache.insert(1, "1");
        cache.insert(2, "2");
        assert_eq!(cache.get(&1), Some(&"1"));
        assert_eq!(cache.get(&2), Some(&"2"));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_put_update() {
        let mut cache = LruSizeCache::new(1);
        cache.insert("1", "1");
        cache.insert("1", "2");
        assert_eq!(cache.get("1"), Some(&"2"));
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_contains_key() {
        let mut cache = LruSizeCache::new(1);
        cache.insert("1", "1");
        assert_eq!(cache.contains_key("1"), true);
    }

    #[test]
    fn test_expire_lru() {
        let mut cache = LruSizeCache::new(8);
        cache.insert("foo1", "bar1");
        cache.insert("foo2", "bar2");
        cache.insert("foo3", "bar3");
        assert!(cache.get("foo1").is_none());
        cache.insert("foo2", "baz2");
        cache.insert("foo4", "bar4");
        assert!(cache.get("foo2").is_some());
        assert!(cache.get("foo3").is_none());
    }

    #[test]
    fn test_pop() {
        let mut cache = LruSizeCache::new(2);
        cache.insert(1, "1");
        cache.insert(2, "2");
        assert_eq!(cache.len(), 2);
        let opt1 = cache.remove(&1);
        assert!(opt1.is_some());
        assert_eq!(opt1.unwrap(), "1");
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_change_capacity() {
        let mut cache = LruSizeCache::new(2);
        assert_eq!(cache.capacity(), 2);
        cache.insert(1, "1");
        cache.insert(2, "2");
        cache.set_capacity(1);
        assert!(cache.get(&1).is_none());
        assert_eq!(cache.capacity(), 1);
    }

    #[test]
    fn test_debug() {
        let mut cache = LruSizeCache::new(3);
        cache.insert(1, "1");
        cache.insert(2, "2");
        cache.insert(3, "3");
        assert_eq!(format!("{:?}", cache), "{3: \"3\", 2: \"2\", 1: \"1\"}");
        cache.insert(2, "4");
        assert_eq!(format!("{:?}", cache), "{2: \"4\", 3: \"3\", 1: \"1\"}");
        cache.insert(6, "6");
        assert_eq!(format!("{:?}", cache), "{6: \"6\", 2: \"4\", 3: \"3\"}");
        cache.get(&3);
        assert_eq!(format!("{:?}", cache), "{3: \"3\", 6: \"6\", 2: \"4\"}");
        cache.set_capacity(2);
        assert_eq!(format!("{:?}", cache), "{3: \"3\", 6: \"6\"}");
    }

    #[test]
    fn test_remove() {
        let mut cache = LruSizeCache::new(3);
        cache.insert(1, "1");
        cache.insert(2, "2");
        cache.insert(3, "3");
        cache.insert(4, "4");
        cache.insert(5, "5");
        cache.remove(&3);
        cache.remove(&4);
        assert!(cache.get(&3).is_none());
        assert!(cache.get(&4).is_none());
        cache.insert(6, "6");
        cache.insert(7, "7");
        cache.insert(8, "8");
        assert!(cache.get(&5).is_none());
        assert_eq!(cache.get(&6), Some(&"6"));
        assert_eq!(cache.get(&7), Some(&"7"));
        assert_eq!(cache.get(&8), Some(&"8"));
    }

    #[test]
    fn test_clear() {
        let mut cache = LruSizeCache::new(2);
        cache.insert(1, "1");
        cache.insert(2, "2");
        cache.clear();
        assert!(cache.get(&1).is_none());
        assert!(cache.get(&2).is_none());
        assert_eq!(format!("{:?}", cache), "{}");
    }

    #[test]
    fn test_iter() {
        let mut cache = LruSizeCache::new(3);
        cache.insert(1, "1");
        cache.insert(2, "2");
        cache.insert(3, "3");
        cache.insert(4, "4");
        cache.insert(5, "5");
        assert_eq!(cache.iter().collect::<Vec<_>>(),
                   [(&3, &"3"), (&4, &"4"), (&5, &"5")]);
        assert_eq!(cache.iter().rev().collect::<Vec<_>>(),
                   [(&5, &"5"), (&4, &"4"), (&3, &"3")]);
    }
}
