/*
 * Stub for _dl_find_object (glibc 2.35+).
 * GCC >= 12's libgcc_eh.a references this symbol for fast unwinding.
 * musl libc does not provide it. Returning -1 tells the unwinder
 * to fall back to the slower (but correct) dl_iterate_phdr path.
 */
struct dl_find_object;
int _dl_find_object(void *address, struct dl_find_object *result) {
    (void)address;
    (void)result;
    return -1;
}
