
from __future__ import annotations

import heapq
from collections import defaultdict
from typing import BinaryIO

EOF = 256  # 1 greater than the possible values for input bytes
DEPTH_LIMIT = 255  # 1 byte
SYMBOL_LIMIT = 257  # Bytes 00000000 -> 11111111, plus our EOF symbol

class Node:
    def __init__(self, symbol: int, left: Node = None, right: Node = None, dummy: bool = False):
        self.left = left
        self.right = right
        self.symbol = symbol
        self.dummy = dummy


class HeapObject():
    def __init__(self, frequency: int, symbol: int, node:Node):
        self.frequency = frequency
        self.symbol = symbol
        self.node = node

    def __lt__(self, other: HeapObject):
        if self.frequency == other.frequency:
            return self.symbol < other.symbol
        return self.frequency < other.frequency

    def __gt__(self, other: HeapObject):
        if self.frequency == other.frequency:
            return self.symbol > other.symbol
        return self.frequency > other.frequency

    def __eq__(self, other: HeapObject):
        return self.frequency == other.frequency and self.symbol == other.symbol

    def __ne__(self, other: HeapObject):
        return not(self == other)


class Code():
    def __init__(self, symbol: int, value: int, depth: int):
        self.symbol = symbol
        self.value = value
        self.depth = depth        

    def __eq__(self, other: Code):
        return self.value == other.value and self.depth == other.depth

    def __ne__(self, other: Code):
        return not(self == other)

    def __hash__(self):
        return hash((self.value, self.depth))


class BitWriter():
    def __init__(self, out: BinaryIO):
        self.out = out
        self.current_byte = 0
        self.bits_written = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def write_bit(self, b, result):
        if b not in (0, 1):
            raise ValueError(f"Got unexpected bit: {b}")

        # Push the bit onto the byte buffer, writing a full byte at a time
        self.current_byte = (self.current_byte << 1) | b
        self.bits_written += 1
        if self.bits_written > 0 and self.bits_written % 8 == 0:
            result.append(self.current_byte)
            self.current_byte = 0

    def write_bits_for_code(self, code: Code, result):
        if code.depth > DEPTH_LIMIT:
            raise ValueError(f"Unsupported output depth: {code.depth} for input symbol: {code.symbol}")
        # Walk the code value for its entire depth, writing each bit
        for i in reversed(range(code.depth)):
            place = code.value & (1 << i)
            if place > 0:
                self.write_bit(1, result)
            else:
                self.write_bit(0, result)
    
    def close(self):
        # If we have unwritten bits, pad with zeros, then write
        pass


class BitReader():
    def __init__(self, input: BinaryIO):
        self.input = input
        self.current_byte = 0
        self.bits_read = 0

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def read_bit(self):
        while self.current_byte != -1:
            if self.bits_read % 8 == 0:
                input_bytes = self.input.read(1)
                if len(input_bytes) < 1:
                    self.current_byte = -1
                    return
                self.current_byte = input_bytes[0]
            
            place = self.current_byte & (1 << (8-(self.bits_read % 8))-1)
            self.bits_read += 1

            if place > 0:
                yield 1
            else:
                yield 0

    def close(self):
        self.current_byte = -1


def compress(bitstream):
    # Indexes 0-255 are for out input file byte frequencies
    # Index 256 will have a frequency of 1, to indicate our EOF marker
    # Because our word length is 1 byte (8 bits), 255 is the max value, leaving symbol 256 available
    frequencies = [0]*SYMBOL_LIMIT
    frequencies[EOF] = 1

    for b in bitstream:
        frequencies[b] += 1

    heap = []
    # Build the heap of symbols with non-zero frequencies
    for i, freq in enumerate(frequencies):
        if freq > 0:
            heapq.heappush(heap, HeapObject(freq, i, Node(i)))

    # Pad until there are at least two items
    for i, freq in enumerate(frequencies):
        if len(heap) >=2:
            break
        if freq == 0:
            heapq.heappush(heap, HeapObject(freq, i, Node(i)))
            
    heapq.heappush(heap, HeapObject(freq, -1, Node(-1, None, None, True)))

    # At this point, if we don't have at least two items, we can't continue
    if len(heap) < 2:
        raise ValueError("Input file has insufficient data to apply encoding.")

    # Loop over the HeapObjects and build out the code tree of InternalNodes and Leafs
    while len(heap) > 1:
        # Pop off two HeapObjects
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
        
        if x.node.dummy:
            x, y = y, x 
        
        z = HeapObject(x.frequency + y.frequency, min(x.symbol, y.symbol), Node(-1, x.node, y.node, x.node.dummy or y.node.dummy))
        heapq.heappush(heap, z)

    # Now, we have a single HeapObject, with our code tree inside
    code_tree = heap[0].node

    codes = set()
    def build_code_list(node, code_value, depth):
        if not node.left and not node.right:
            codes.add(Code(node.symbol, code_value, depth))
        if node.left:
            build_code_list(node.left, (code_value<<1), depth+1)
        if node.right:
            build_code_list(node.right, (code_value<<1)+1, depth+1)

    build_code_list(code_tree, 0, 0)


    canonical_codes = sorted(codes, key=lambda x: (x.depth, x.symbol))
    current_depth = canonical_codes[0].depth

    codebook_depths = [0]*SYMBOL_LIMIT
    next_code_value = 0
    for code in canonical_codes:
        code.value = next_code_value
        if code.depth > current_depth:
            code.value = code.value << (code.depth - current_depth)
            current_depth = code.depth
        next_code_value = code.value + 1

        codebook_depths[code.symbol] = current_depth

    # Now fill out the code book array for easy index access symbol (0-256) -> code
    canonical_codebook = [None]*SYMBOL_LIMIT
    for code in canonical_codes:
        canonical_codebook[code.symbol] = code

    result = bytearray()
    # The first 257 bytes of the file are the codebook depths
    for depth in codebook_depths:
        result.append(depth)

    with BitWriter("aaa") as bit_writer:  # Use a context manager to automatically call .close() at the end
        # We've already read the file once to build the codebook
        # Seek to the begginging to read again and encode
        
        for b in bitstream:  # read 1 byte at a time
            code = canonical_codebook[b]  # translate to our huffman code
            bit_writer.write_bits_for_code(code, result)  # write the bits for that code
    
        # Now we're at the end of the input file, so write our EOF symbol
        code = canonical_codebook[EOF]
        bit_writer.write_bits_for_code(code, result)

        
        return result

def decompress(input_file_path: str, output_file_path: str):
    # The depths of each symbol, Depth -> [Sym_1, Sym_2...]
    codebook_depths = defaultdict(list)

    # A dictionary of Code -> Symbol
    # Codes are hashed by Value and Depth
    codebook = {}

    with open(input_file_path, "rb") as input_file, open(output_file_path, "wb") as output_file:
        # Read the codebook_depth header
        for symbol in range(SYMBOL_LIMIT):  
            input_bytes = input_file.read(1)
            depth = input_bytes[0]
            codebook_depths[depth].append(symbol)
        
        # Rebuild the symbol table
        # Again: https://en.wikipedia.org/wiki/Canonical_Huffman_code#Algorithm
        current_depth = None
        next_code_value = 0
        for depth in range(DEPTH_LIMIT):
            symbols_at_depth = codebook_depths[depth]
            if depth and symbols_at_depth:
                if not current_depth:
                    current_depth = depth
                if depth > current_depth:
                    next_code_value = next_code_value << (depth - current_depth)
                    current_depth = depth
                for symbol in symbols_at_depth:
                    code = Code(symbol, next_code_value, depth)
                    codebook[code] = symbol
                    next_code_value = next_code_value + 1
        
        # We now have a codebook to translate a Code's Value and Depth back to it's uncompressed Symbol
        # Read in the compressed file, yielding one bit at a time
        with BitReader(input_file) as bit_reader:
            code_candidate = Code(0, 0, 0)  # A dummy code that will attempt to lookup on value and depth
            for bit in bit_reader.read_bit():
                code_candidate.depth += 1
                if bit == 0:
                    code_candidate.value = (code_candidate.value<<1)
                else:
                    code_candidate.value = (code_candidate.value<<1)+1
                # When we find a translation
                if code_candidate in codebook:  # This works because Code hashing and equality ignores symbol
                    if codebook[code_candidate] == EOF:  # Done writing
                        break
                    # We have the original Symbol from the stream, write it to the output
                    output_file.write(bytes([codebook[code_candidate]]))
                    # And prepare a new candidate for the next Symbol
                    code_candidate = Code(0, 0, 0)    
        